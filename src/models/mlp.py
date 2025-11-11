import typing as tp

import torch
from torch import nn


class MLP(nn.Module):
    """Standard Multi-Layer Perceptron for image classification."""

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_classes: int,
        hidden_dims: tp.List[int] = [512, 256],
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        input_dim = img_size * img_size * in_channels

        layers: tp.List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)  # flatten image to vector
        return self.network(x)


class DeepMLP(nn.Module):
    """Deeper variant of MLP with additional hidden layers."""

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_classes: int,
        hidden_dims: tp.List[int] = [1024, 512, 256, 128],
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        input_dim = img_size * img_size * in_channels

        layers: tp.List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)  # flatten image to vector
        return self.network(x)

