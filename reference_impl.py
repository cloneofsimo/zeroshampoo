import math

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb


def get_left_right_shape(param):
    if param.ndim == 1:
        return (param.shape[0], 1)
    else:
        return (np.prod(param.shape[:-1]), param.shape[-1])


def matrix_pth_power_via_eigendecompsition(mat, p=-1 / 4):
    """
    Compute the matrix p-th root of a matrix using eigendecomposition.
    """
    eigvals, eigvecs = torch.linalg.eigh(mat)
    mineig = min(eigvals.min().item(), 0)

    eigvals = eigvals - mineig + 1e-8
    eigvals = eigvals**p

    return eigvecs @ torch.diag(eigvals) @ eigvecs.t()


class ShampooWithAdamGraftingOptimizer:
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        shampoo_eps=1e-8,
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        precondition_frequency=None,
        start_preconditioning=4,
        independent_weight_decay=True,
        weight_decay=0.001,
        device=None,
        dtype=None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= shampoo_eps:
            raise ValueError(f"Invalid epsilon value: {shampoo_eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

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
        self.device = device
        self.dtype = dtype

        self.state = {}
        self._init_state()

    def _init_state(self):
        for param in self.params:
            p_device = self.device if self.device is not None else param.device
            p_dtype = self.dtype if self.dtype is not None else param.dtype

            state = self.state[param] = {}
            state["step"] = 0
            state["m_adam"] = torch.zeros_like(param, device=p_device, dtype=p_dtype)
            state["v_adam"] = torch.zeros_like(param, device=p_device, dtype=p_dtype)

            left_shape, right_shape = get_left_right_shape(param)
            state["left_preconditioner_accum"] = self.shampoo_eps * torch.eye(
                left_shape, device=p_device, dtype=p_dtype
            )
            state["right_preconditioner_accum"] = self.shampoo_eps * torch.eye(
                right_shape, device=p_device, dtype=p_dtype
            )
            state["left_preconditioner"] = torch.eye(
                left_shape, device=p_device, dtype=p_dtype
            )
            state["right_preconditioner"] = torch.eye(
                right_shape, device=p_device, dtype=p_dtype
            )

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    @torch.no_grad()
    def step(self):
        for param in self.params:
            if param.grad is None:
                continue

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
            left_shape, right_shape = get_left_right_shape(param)
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
                        matrix_pth_power_via_eigendecompsition(
                            state["left_preconditioner_accum"], p=-1 / 4
                        )
                    )
                    state["right_preconditioner"] = (
                        matrix_pth_power_via_eigendecompsition(
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

                # print(fnorm_of_adam_update_dir, fnorm_of_shampoo_update_dir)

            else:
                update_dir = m_hat / (torch.sqrt(v_hat) + self.adam_eps)
                update_dir = update_dir.view((left_shape, right_shape))

            param.data.add_(update_dir.view(grad_shape), alpha=-self.lr)


class Net(nn.Module):
    def __init__(self, width=32):
        super(Net, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(28 * 28, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 10),
        )

        self.mlp[4].weight.data.fill_(0)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


# Evaluation function
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
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return accuracy, test_loss


@click.command()
@click.option("--width", default=128, help="Width of the hidden layers")
@click.option("--lr", default=1e-3, help="Learning rate")
@click.option("--epochs", default=4, help="Number of epochs")
@click.option("--shampoo", is_flag=True, help="Use Shampoo optimizer")
def main(width=128, lr=1e-3, epochs=2, shampoo=False):

    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set hyperparameters
    batch_size = 256
    test_batch_size = 1000

    wandb.init(
        project="shampoo",
        entity="simo",
        name=f"mnist_shampoo_{width}_{lr}_{epochs}_{batch_size}_{test_batch_size}",
        config={
            "width": width,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "test_batch_size": test_batch_size,
            "shampoo": 0.0 if not shampoo else 1.0,
        },
        tags=["sweep_final_zero"],
    )

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("../data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # Initialize the model and optimizer
    model = Net(width).to(device)
    if shampoo:
        optimizer = ShampooWithAdamGraftingOptimizer(
            model.parameters(), lr=lr / width, start_preconditioning=20
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr / width)

    # Training and evaluation loop
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy, test_loss = test(model, device, test_loader)
        wandb.log({"accuracy": accuracy, "test_loss": test_loss})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Best test accuracy: {best_accuracy:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    main()
