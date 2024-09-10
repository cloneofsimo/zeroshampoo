import numpy as np
import torch
import torch.distributed as dist


class ZeroShampooWithAdamGraftingOptimizer:
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        shampoo_eps=1e-6,
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        precondition_frequency=None,
        start_preconditioning=4,
        independent_weight_decay=True,
        weight_decay=0.001,
        device=None,
        dtype=None,
        block_size=128,
    ):
        if isinstance(params, (list, tuple)) and isinstance(params[0], dict):
            self.param_groups = params
        else:
            self.param_groups = [{"params": list(params)}]

        self.defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_eps=shampoo_eps,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            precondition_frequency=precondition_frequency or start_preconditioning,
            start_preconditioning=start_preconditioning,
            independent_weight_decay=independent_weight_decay,
            weight_decay=weight_decay,
        )
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype or torch.float32
        self.state = {}
        self.block_size = block_size

        # Distributed training setup
        try:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.is_distributed = True
        except Exception as e:
            print(
                "Distributed training not initialized, setting rank and world size to 0"
            )
            self.rank = 0
            self.world_size = 1
            self.is_distributed = False

        self.param_stats = {}
        self._make_lookup_and_enumeratables()
        self._init_state()

    @torch.no_grad()
    def _make_lookup_and_enumeratables(self):
        self.lookup = {}
        self.enumeratables = []
        global_counter = 0
        total_params = 0

        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    name = f"param"
                    s1, s2 = self._get_left_right_shape(param)
                    for i1 in range(0, s1, self.block_size):
                        i1r = min(i1 + self.block_size, s1)
                        for i2 in range(0, s2, self.block_size):
                            i2r = min(i2 + self.block_size, s2)
                            block_name = (
                                f"{name}_{global_counter}_{i1}_{i1r}_{i2}_{i2r}"
                            )
                            self.enumeratables.append(
                                (
                                    global_counter,
                                    block_name,
                                    param,
                                    (s1, s2),
                                    (i1, i1r),
                                    (i2, i2r),
                                    group,
                                )
                            )
                            total_params += (i1r - i1) * (i2r - i2)
                            if param not in self.param_stats:
                                self.param_stats[param] = []

                            self.param_stats[param].append(
                                (i1, i1r, i2, i2r, s1, s2, block_name)
                            )

                    global_counter += 1

            # make default
            for k, v in self.defaults.items():
                group[k] = v

        total_param_in_model = 0
        for group in self.param_groups:
            for param in group["params"]:
                total_param_in_model += param.numel()

        assert (
            total_params == total_param_in_model
        ), f"Total params: {total_params} != {total_param_in_model}"

    def _enumerate_sharded_params(self):
        for (
            global_counter,
            block_name,
            param,
            (s1, s2),
            (i1, i1r),
            (i2, i2r),
            group,
        ) in self.enumeratables:
            if global_counter % self.world_size != self.rank:
                continue
            yield block_name, param, (s1, s2), (i1, i1r), (i2, i2r), group

    def _init_state(self):
        for (
            block_name,
            param,
            (s1, s2),
            (i1, i1r),
            (i2, i2r),
            group,
        ) in self._enumerate_sharded_params():
            block_param = param.view(s1, s2)[i1:i1r, i2:i2r]
            print(
                f"Rank {self.rank} is managing parameter {block_name}, shape: {block_param.shape}, dtype: {block_param.dtype}, range {i1}:{i1r}, {i2}:{i2r}"
            )
            assert (
                self.state.get(block_name, None) is None
            ), f"State for {block_name} already exists"
            self.state[block_name] = {}
            state = self.state[block_name]
            state["step"] = 0
            state["m_adam"] = torch.zeros_like(
                block_param, device=self.device, dtype=self.dtype
            )
            state["v_adam"] = torch.zeros_like(
                block_param, device=self.device, dtype=self.dtype
            )
            state["left_preconditioner_accum"] = group["shampoo_eps"] * torch.eye(
                i1r - i1, device=self.device, dtype=self.dtype
            )
            state["right_preconditioner_accum"] = group["shampoo_eps"] * torch.eye(
                i2r - i2, device=self.device, dtype=self.dtype
            )
            state["left_preconditioner"] = None
            state["right_preconditioner"] = None

    def _get_left_right_shape(self, param):
        if param.ndim == 1:
            return (param.shape[0], 1)
        else:
            return (np.prod(param.shape[:-1]), param.shape[-1])

    def zero_grad(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()

    @torch.no_grad()
    def step(self):
        self._reduce_gradients()

        for (
            block_name,
            param,
            (s1, s2),
            (i1, i1r),
            (i2, i2r),
            group,
        ) in self._enumerate_sharded_params():
            grad = param.grad
            assert grad is not None, f"Gradient is None for {block_name}"
            state = self.state[block_name]

            block_param = param.view(s1, s2)[i1:i1r, i2:i2r]
            block_grad = grad.view(s1, s2)[i1:i1r, i2:i2r]

            assert block_param.shape == block_grad.shape, (
                block_param.shape,
                block_grad.shape,
            )

            left_shape, right_shape = block_param.shape

            # Update step count
            state["step"] += 1

            # Get group-specific hyperparameters
            lr = group["lr"]

            weight_decay = group["weight_decay"]
            independent_weight_decay = group["independent_weight_decay"]
            shampoo_beta1, shampoo_beta2 = group["betas"]
            adam_beta1, adam_beta2 = group["adam_betas"]
            adam_eps = group["adam_eps"]
            start_preconditioning = group["start_preconditioning"]
            precondition_frequency = group["precondition_frequency"]

            # Perform stepweight decay
            if independent_weight_decay:
                block_param.data.mul_(1 - lr * weight_decay)

            block_grad_shape = block_grad.shape

            # Update preconditioners
            state["left_preconditioner_accum"].mul_(shampoo_beta1).add_(
                block_grad @ block_grad.t(), alpha=1 - shampoo_beta1
            )
            state["right_preconditioner_accum"].mul_(shampoo_beta1).add_(
                block_grad.t() @ block_grad, alpha=1 - shampoo_beta1
            )

            # Update Adam state
            state["m_adam"].mul_(adam_beta1).add_(block_grad, alpha=1 - adam_beta1)
            state["v_adam"].mul_(adam_beta2).addcmul_(
                block_grad, block_grad, value=1 - adam_beta2
            )

            m_hat = state["m_adam"] / (1 - adam_beta1 ** state["step"])
            v_hat = state["v_adam"] / (1 - adam_beta2 ** state["step"])
            adam_update_dir = m_hat / (torch.sqrt(v_hat) + adam_eps)

            if state["step"] >= start_preconditioning:
                if state["step"] % precondition_frequency == 0:
                    state[
                        "left_preconditioner"
                    ] = self._matrix_pth_power_via_eigendecompsition(
                        state["left_preconditioner_accum"], p=-1 / 4
                    )
                    state[
                        "right_preconditioner"
                    ] = self._matrix_pth_power_via_eigendecompsition(
                        state["right_preconditioner_accum"], p=-1 / 4
                    )

                fnorm_of_adam_update_dir = torch.linalg.norm(adam_update_dir)
                grad_momentum = state["m_adam"]

                shampoo_update_dir = (
                    state["left_preconditioner"]
                    @ grad_momentum
                    @ state["right_preconditioner"]
                )

                fnorm_of_shampoo_update_dir = torch.linalg.norm(shampoo_update_dir)

                update_dir = (
                    fnorm_of_adam_update_dir
                    * shampoo_update_dir
                    / fnorm_of_shampoo_update_dir
                )
            else:
                update_dir = adam_update_dir

            assert update_dir.shape == block_param.shape
            assert update_dir.shape == block_grad.shape

            param.view(s1, s2)[i1:i1r, i2:i2r].data.add_(update_dir, alpha=-lr)

        self._sync_params()

    def _check_momentum_and_variance(self):
        num_total_params = 0
        # iterate over all params
        for group in self.param_groups:
            for param in group["params"]:
                num_total_params += param.numel()

        num_non_zero_params = 0
        for (
            block_name,
            param,
            (s1, s2),
            (i1, i1r),
            (i2, i2r),
            group,
        ) in self._enumerate_sharded_params():
            state = self.state[block_name]
            # check if the values are very-close to non-zero or not
            assert not torch.allclose(
                state["m_adam"], torch.zeros_like(state["m_adam"]), atol=1e-8
            ), f"Momentum is zero for {block_name}: average var: {state['m_adam'].abs().mean()}, state: {state['m_adam']}"
            assert not torch.allclose(
                state["v_adam"], torch.zeros_like(state["v_adam"]), atol=1e-8
            ), f"Variance is zero for {block_name}: average var: {state['v_adam'].abs().mean()}, state: {state['v_adam']}"
            num_non_zero_params += (i1r - i1) * (i2r - i2)

        assert (
            num_non_zero_params == num_total_params
        ), f"Num non-zero params: {num_non_zero_params} != {num_total_params}"
        print("All momentum and variance are non-zero")

    def build_global_state_for_debug_purposes(self):
        self.global_state = {}
        for (
            global_counter,
            block_name,
            param,
            (s1, s2),
            (i1, i1r),
            (i2, i2r),
            group,
        ) in self.enumeratables:
            if global_counter % self.world_size != self.rank:
                continue

            if param not in self.global_state:
                self.global_state[param] = {}
            # make exp_avg, exp_avg_sq
            if "exp_avg" not in self.global_state[param]:
                self.global_state[param]["exp_avg"] = torch.ones_like(param.data).view(
                    s1, s2
                )
            if "exp_avg_sq" not in self.global_state[param]:
                self.global_state[param]["exp_avg_sq"] = torch.ones_like(
                    param.data
                ).view(s1, s2)

            print(f"Doing {block_name}, {i1}:{i1r}, {i2}:{i2r}")
            assert self.state[block_name]["m_adam"].shape == (i1r - i1, i2r - i2)
            # fill in
            self.global_state[param]["exp_avg"][i1:i1r, i2:i2r] = self.state[
                block_name
            ]["m_adam"]
            self.global_state[param]["exp_avg_sq"][i1:i1r, i2:i2r] = self.state[
                block_name
            ]["v_adam"]

    @torch.no_grad()
    def _matrix_pth_power_via_eigendecompsition(self, mat, p=-1 / 4):
        try:
            eigvals, eigvecs = torch.linalg.eigh(mat)
        except Exception as e:
            print("RuntimeError in _matrix_pth_power_via_eigendecompsition")
            print("mat", mat)
            print("p", p)
            print("trace", mat.trace().item())
            print("rank", self.rank)

            raise

        mineig = min(eigvals.min().item(), 0)

        eigvals = eigvals - mineig + 1e-8
        eigvals = eigvals**p

        return eigvecs @ torch.diag(eigvals) @ eigvecs.t()

    @torch.no_grad()
    def _sync_params(self):
        if not self.is_distributed:
            return
        did_broadcast_list = set()
        for (
            global_counter,
            block_name,
            param,
            (s1, s2),
            (i1, i1r),
            (
                i2,
                i2r,
            ),
            group,
        ) in self.enumeratables:
            if global_counter in did_broadcast_list:
                continue

            if global_counter % self.world_size == self.rank:
                dist.broadcast(param.data, src=self.rank)
            else:
                dist.broadcast(param.data, src=global_counter % self.world_size)

            did_broadcast_list.add(global_counter)

    @torch.no_grad()
    def _reduce_gradients(self):
        if not self.is_distributed:
            return
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
