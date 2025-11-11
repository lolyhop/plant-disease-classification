import typing as tp

import torch
from torch import nn, optim

OPTIMIZER_REGISTRY: tp.Dict[str, tp.Type[optim.Optimizer]] = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
}


def build_optimizer(
    model: nn.Module,
    name: str,
    lr: float,
    **kwargs: tp.Any,
) -> optim.Optimizer:
    """
    Optimizer factory.
    """
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}")

    # Convert string numeric values to floats (YAML sometimes parses scientific notation as strings)
    processed_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            # Try to convert string numbers to float
            try:
                processed_kwargs[key] = float(value)
            except (ValueError, TypeError):
                processed_kwargs[key] = value
        elif isinstance(value, list) and key == "betas":
            # Ensure betas are floats
            processed_kwargs[key] = [float(v) if isinstance(v, (str, int)) else v for v in value]
        else:
            processed_kwargs[key] = value

    OptimClass = OPTIMIZER_REGISTRY[name]
    return OptimClass(model.parameters(), lr=float(lr), **processed_kwargs)


def compute_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm**0.5


def clip_gradients(model: nn.Module, max_norm: float) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
