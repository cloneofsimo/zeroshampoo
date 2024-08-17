import math

import numpy as np
import torch
import torch.distributed as dist

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
    ):
        self.params = list(params)
        self.lr = lr
        self.shampoo_beta1, self.shampoo_beta2 = betas
        self.shampoo_eps = shampoo_eps
        self.adam_beta1, self.adam_beta2 = adam_betas
        self.adam_eps = adam_eps
        self.precondition_frequency = precondition_frequency or start_preconditioning
        self.start_preconditioning = start_preconditioning
        self.independent_weight_decay = independent_weight_decay
        self.weight_decay = weight_decay
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype or torch.float32

        self.state = {}
        # Distributed training setup
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self._init_state()

    def _init_state(self):
        for i, param in enumerate(self.params):
            if i % self.world_size != self.rank:
                continue  # Skip parameters not managed by this rank
            print(f"Rank {self.rank} is managing parameter {i}, shape: {param.shape}")
            state = self.state[param] = {}
            state["step"] = 0
            state["m_adam"] = torch.zeros_like(
                param, device=self.device, dtype=self.dtype
            )
            state["v_adam"] = torch.zeros_like(
                param, device=self.device, dtype=self.dtype
            )

            left_shape, right_shape = self._get_left_right_shape(param)
            state["left_preconditioner_accum"] = self.shampoo_eps * torch.eye(
                left_shape, device=self.device, dtype=self.dtype
            )
            state["right_preconditioner_accum"] = self.shampoo_eps * torch.eye(
                right_shape, device=self.device, dtype=self.dtype
            )
            state["left_preconditioner"] = torch.eye(
                left_shape, device=self.device, dtype=self.dtype
            )
            state["right_preconditioner"] = torch.eye(
                right_shape, device=self.device, dtype=self.dtype
            )

    def _get_left_right_shape(self, param):
        if param.ndim == 1:
            return (param.shape[0], 1)
        else:
            return (np.prod(param.shape[:-1]), param.shape[-1])

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    @torch.no_grad()
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            if i % self.world_size != self.rank:
                continue  # Skip parameters not managed by this rank

            grad = param.grad
            state = self.state[param]

            # Update step count
            state["step"] += 1

            # Perform stepweight decay
            if self.independent_weight_decay:
                param.data.mul_(1 - self.lr * self.weight_decay)

            # If doing "add to input" weight decay, do it here
            if not self.independent_weight_decay:
                grad = grad.add(param, alpha=self.weight_decay)

            # Get shapes for preconditioners
            grad_shape = grad.shape
            left_shape, right_shape = self._get_left_right_shape(param)
            grad_rs = grad.view(left_shape, right_shape)

            # Update preconditioners
            state["left_preconditioner_accum"].mul_(self.shampoo_beta1).add_(
                grad_rs @ grad_rs.t(), alpha=1 - self.shampoo_beta1
            )
            state["right_preconditioner_accum"].mul_(self.shampoo_beta1).add_(
                grad_rs.t() @ grad_rs, alpha=1 - self.shampoo_beta1
            )

            # Update Adam state
            state["m_adam"].mul_(self.adam_beta1).add_(grad, alpha=1 - self.adam_beta1)
            state["v_adam"].mul_(self.adam_beta2).addcmul_(
                grad, grad, value=1 - self.adam_beta2
            )

            m_hat = state["m_adam"] / (1 - self.adam_beta1 ** state["step"])
            v_hat = state["v_adam"] / (1 - self.adam_beta2 ** state["step"])

            if state["step"] >= self.start_preconditioning:
                if state["step"] % self.precondition_frequency == 0:
                    state["left_preconditioner"] = (
                        self._matrix_pth_power_via_eigendecompsition(
                            state["left_preconditioner_accum"], p=-1 / 4
                        )
                    )
                    state["right_preconditioner"] = (
                        self._matrix_pth_power_via_eigendecompsition(
                            state["right_preconditioner_accum"], p=-1 / 4
                        )
                    )

                adam_update_dir = m_hat / (torch.sqrt(v_hat) + self.adam_eps)
                fnorm_of_adam_update_dir = torch.linalg.norm(adam_update_dir)

                shampoo_update_dir = (
                    state["left_preconditioner"]
                    @ grad_rs
                    @ state["right_preconditioner"]
                )

                fnorm_of_shampoo_update_dir = torch.linalg.norm(shampoo_update_dir)

                update_dir = (
                    fnorm_of_adam_update_dir
                    * shampoo_update_dir
                    / fnorm_of_shampoo_update_dir
                )
            else:
                update_dir = m_hat / (torch.sqrt(v_hat) + self.adam_eps)
                update_dir = update_dir.view((left_shape, right_shape))

            param.data.add_(update_dir.view(grad_shape), alpha=-self.lr)

        # Synchronize updated parameters across GPUs
        self._sync_params()

    def _matrix_pth_power_via_eigendecompsition(self, mat, p=-1 / 4):
        eigvals, eigvecs = torch.linalg.eigh(mat)
        mineig = min(eigvals.min().item(), 0)

        eigvals = eigvals - mineig + 1e-8
        eigvals = eigvals**p

        return eigvecs @ torch.diag(eigvals) @ eigvecs.t()

    def _sync_params(self):
        for i, param in enumerate(self.params):
            if i % self.world_size == self.rank:
                # Broadcast parameters from the responsible rank to all others
                dist.broadcast(param.data, src=self.rank)
            else:
                # Receive parameters from the responsible rank
                dist.broadcast(param.data, src=i % self.world_size)

    def reduce_gradients(self):
        for param in self.params:
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size


import os
import time

import click
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

import wandb

TOKEN_LENGTH = 1024


class Net(nn.Module):
    def __init__(self, width=32):
        super(Net, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(28 * 28, width),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.ReLU(),
            nn.Linear(width, 10),
        )

        # self.mlp[4].weight.data.fill_(0)

    @torch.compile()
    def forward(self, x):
        x = x.view(-1, 1, 28 * 28)
        x = x.repeat(1, TOKEN_LENGTH, 1)
        x = self.mlp(x)
        x = x.mean(dim=1)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.reduce_gradients()
        optimizer.step()

        if batch_idx % 100 == 0 and dist.get_rank() == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data) * dist.get_world_size()}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
            wandb.log({"train_loss": loss.item(), "epoch": epoch, "batch": batch_idx})

    print(batch_idx)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    if dist.get_rank() == 0:
        print(
            f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
        )
        wandb.log({"test_loss": test_loss, "accuracy": accuracy})

    return accuracy, test_loss


from pathlib import Path

from torch.profiler import ProfilerActivity, profile, record_function


@click.command()
@click.option("--width", default=128, help="Width of the hidden layers")
@click.option("--lr", default=1e-3, help="Learning rate")
@click.option("--epochs", default=4, help="Number of epochs")
@click.option("--shampoo", is_flag=True, help="Use Shampoo optimizer")
@click.option("--do_profile", is_flag=True, help="Profile the training")
def main(width=128, lr=1e-3, epochs=2, shampoo=True, do_profile=False):
    # Initialize distributed environment
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Set hyperparameters
    per_device_batch_size = 64
    test_batch_size = 1000

    # (1024 x 4096) @ (4096 x 4096)

    if rank == 0:
        wandb.init(
            project="zero-shampoo",
            entity="simo",
            name=f"mnist_zero_shampoo_{width}_{lr}_{epochs}_{per_device_batch_size}_{test_batch_size}",
            config={
                "width": width,
                "lr": lr,
                "epochs": epochs,
                "batch_size": per_device_batch_size * world_size,
                "test_batch_size": test_batch_size,
                "shampoo": shampoo,
                "world_size": world_size,
            },
            tags=["zero_shampoo"],
        )

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("../data", train=False, transform=transform)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        sampler=train_sampler,
        num_workers=8,
    )
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # Initialize the model and optimizer
    model = Net(width).to(device)

    num_params = sum(p.numel() for p in model.parameters())  # 187M
    flop_for_fwd_bwd = 6 * num_params * len(train_dataset) * TOKEN_LENGTH
    theoretical_flops = world_size * 500 * 10**12  # 500 TFLOPS per device

    # model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if shampoo:
        optimizer = ZeroShampooWithAdamGraftingOptimizer(
            model.parameters(), lr=lr / width, start_preconditioning=20
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr / width)

    # Training and evaluation loop
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        t0 = time.time()

        if dist.get_rank() == 0 and do_profile:
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            )
            prof.__enter__()

        train(model, device, train_loader, optimizer, epoch)

        if dist.get_rank() == 0 and do_profile:
            prof.__exit__(None, None, None)

            print("saving profile to", Path("./checkpoint") / "trace.json")
            prof.export_chrome_trace(str(Path("./checkpoint") / "trace.json"))

        T = time.time() - t0
        if dist.get_rank() == 0:
            print(f"Epoch {epoch} took {T:.2f} seconds")
            print(f"MFU: {100 * flop_for_fwd_bwd / T / theoretical_flops:.2f}%")
            # 6 * 187805706 * 10 * 8 * 64 * 512 / 1.61 / (8 * 500 * 10^12) # 0.45
            # 6 * N * (10 * 8 * 64 * 512) / (1.61 * 8 * 980 * 10^12)
            print(f"TFLOPS: {theoretical_flops / 10**12}")
            print(f"TFlop required for one epoch: {flop_for_fwd_bwd / 10**12}")
            print(f"num_params: {num_params}")

        accuracy, test_loss = test(model, device, test_loader)

    if rank == 0:
        print(f"Best test accuracy: {best_accuracy:.2f}%")
        wandb.finish()


if __name__ == "__main__":
    main()
