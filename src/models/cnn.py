import typing as tp

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: tp.Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # Main path: two 3x3 conv layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (downsample if needed)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Residual connection
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet-50/101/152."""

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: tp.Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # Main path: 1x1 -> 3x3 -> 1x1 conv layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Residual connection
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Custom ResNet implementation with residual connections."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "resnet18",
        pretrained: bool = False,  
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        
        # Define layer configurations for different ResNet variants
        if model_name == "resnet18":
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif model_name == "resnet34":
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif model_name == "resnet50":
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif model_name == "resnet101":
            block = Bottleneck
            layers = [3, 4, 23, 3]
        else:
            raise ValueError(f"Unknown ResNet model: {model_name}")

        self.in_channels = 64

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 * block.expansion, num_classes),
        )

        self._init_weights()

    def _make_layer(
        self,
        block: tp.Type[nn.Module],
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers: tp.List[nn.Module] = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling and classification
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


class DenseLayer(nn.Module):
    """Dense layer with batch normalization and ReLU."""

    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4) -> None:
        super().__init__()
        # Bottleneck: 1x1 conv to reduce channels
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)
        
        # 3x3 conv
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(x)
        out = torch.relu_(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu_(out)
        out = self.conv2(out)
        return out


class DenseBlock(nn.Module):
    """Dense block with multiple dense layers."""

    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, bn_size: int = 4) -> None:
        super().__init__()
        layers: tp.List[nn.Module] = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate all previous feature maps with current output
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)


class Transition(nn.Module):
    """Transition layer between dense blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = torch.relu_(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    """Custom DenseNet implementation with dense connections."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "densenet121",
        pretrained: bool = False,  
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        
        # Define configurations for different DenseNet variants
        if model_name == "densenet121":
            growth_rate = 32
            block_config = [6, 12, 24, 16]
        elif model_name == "densenet169":
            growth_rate = 32
            block_config = [6, 12, 32, 32]
        elif model_name == "densenet201":
            growth_rate = 32
            block_config = [6, 12, 48, 32]
        else:
            raise ValueError(f"Unknown DenseNet model: {model_name}")

        num_init_features = 64
        bn_size = 4

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate, bn_size)
            self.features.add_module(f"denseblock{i+1}", block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                # Add transition layer (except after last block)
                trans = Transition(num_features, num_features // 2)
                self.features.add_module(f"transition{i+1}", trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = torch.relu_(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class EfficientNet(nn.Module):
    """EfficientNet model for plant disease classification."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if model_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        elif model_name == "efficientnet_b1":
            backbone = models.efficientnet_b1(weights="DEFAULT" if pretrained else None)
        elif model_name == "efficientnet_b2":
            backbone = models.efficientnet_b2(weights="DEFAULT" if pretrained else None)
        elif model_name == "efficientnet_b3":
            backbone = models.efficientnet_b3(weights="DEFAULT" if pretrained else None)
        elif model_name == "efficientnet_b4":
            backbone = models.efficientnet_b4(weights="DEFAULT" if pretrained else None)
        else:
            raise ValueError(f"Unknown EfficientNet model: {model_name}")

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone.classifier[1].in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)


