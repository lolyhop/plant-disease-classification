import typing as tp

import torch
import torch.nn.functional as F
from torch import nn


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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block."""

    def __init__(self, in_channels: int, se_ratio: float = 0.25) -> None:
        super().__init__()
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion_ratio: int = 6,
        se_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        expanded_channels = in_channels * expansion_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        # Expansion phase (1x1 conv)
        if expansion_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
            self.expand_bn = nn.BatchNorm2d(expanded_channels)
        else:
            self.expand_conv = None

        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv2d(
            expanded_channels if expansion_ratio != 1 else in_channels,
            expanded_channels if expansion_ratio != 1 else in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=expanded_channels if expansion_ratio != 1 else in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels if expansion_ratio != 1 else in_channels)

        # SE attention
        self.se = SEBlock(expanded_channels if expansion_ratio != 1 else in_channels, se_ratio)

        # Projection phase (1x1 conv)
        self.project_conv = nn.Conv2d(
            expanded_channels if expansion_ratio != 1 else in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.project_bn = nn.BatchNorm2d(out_channels)

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Expansion
        if self.expand_conv is not None:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.activation(x)

        # Depthwise
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.activation(x)

        # SE attention
        x = self.se(x)

        # Projection
        x = self.project_conv(x)
        x = self.project_bn(x)

        # Residual connection
        if self.use_residual:
            x = x + identity

        return x


class EfficientNet(nn.Module):
    """Custom EfficientNet implementation with MBConv blocks and SE attention."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "efficientnet_b0",
        pretrained: bool = False,  # Not used for custom implementation
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # EfficientNet-B0 configuration: (expansion_ratio, out_channels, num_blocks, stride, kernel_size, se_ratio)
        if model_name == "efficientnet_b0":
            width_multiplier = 1.0
            depth_multiplier = 1.0
            dropout_rate = 0.2
            blocks_config = [
                (1, 16, 1, 1, 3, 0.25),   # MBConv1
                (6, 24, 2, 2, 3, 0.25),   # MBConv6
                (6, 40, 2, 2, 5, 0.25),   # MBConv6
                (6, 80, 3, 2, 3, 0.25),   # MBConv6
                (6, 112, 3, 1, 5, 0.25),  # MBConv6
                (6, 192, 4, 2, 5, 0.25),  # MBConv6
                (6, 320, 1, 1, 3, 0.25),  # MBConv6
            ]
        elif model_name == "efficientnet_b1":
            width_multiplier = 1.0
            depth_multiplier = 1.1
            dropout_rate = 0.2
            blocks_config = [
                (1, 16, 1, 1, 3, 0.25),
                (6, 24, 2, 2, 3, 0.25),
                (6, 40, 2, 2, 5, 0.25),
                (6, 80, 3, 2, 3, 0.25),
                (6, 112, 3, 1, 5, 0.25),
                (6, 192, 4, 2, 5, 0.25),
                (6, 320, 1, 1, 3, 0.25),
            ]
        elif model_name == "efficientnet_b2":
            width_multiplier = 1.1
            depth_multiplier = 1.2
            dropout_rate = 0.3
            blocks_config = [
                (1, 16, 1, 1, 3, 0.25),
                (6, 24, 2, 2, 3, 0.25),
                (6, 40, 2, 2, 5, 0.25),
                (6, 80, 3, 2, 3, 0.25),
                (6, 112, 3, 1, 5, 0.25),
                (6, 192, 4, 2, 5, 0.25),
                (6, 320, 1, 1, 3, 0.25),
            ]
        elif model_name == "efficientnet_b3":
            width_multiplier = 1.2
            depth_multiplier = 1.4
            dropout_rate = 0.3
            blocks_config = [
                (1, 16, 1, 1, 3, 0.25),
                (6, 24, 2, 2, 3, 0.25),
                (6, 40, 2, 2, 5, 0.25),
                (6, 80, 3, 2, 3, 0.25),
                (6, 112, 3, 1, 5, 0.25),
                (6, 192, 4, 2, 5, 0.25),
                (6, 320, 1, 1, 3, 0.25),
            ]
        elif model_name == "efficientnet_b4":
            width_multiplier = 1.4
            depth_multiplier = 1.8
            dropout_rate = 0.4
            blocks_config = [
                (1, 16, 1, 1, 3, 0.25),
                (6, 24, 2, 2, 3, 0.25),
                (6, 40, 2, 2, 5, 0.25),
                (6, 80, 3, 2, 3, 0.25),
                (6, 112, 3, 1, 5, 0.25),
                (6, 192, 4, 2, 5, 0.25),
                (6, 320, 1, 1, 3, 0.25),
            ]
        else:
            raise ValueError(f"Unknown EfficientNet model: {model_name}")

        # Calculate channels with width multiplier
        def round_channels(channels: int) -> int:
            return int(channels * width_multiplier)

        # Initial stem
        stem_channels = round_channels(32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )

        # Build MBConv blocks
        self.blocks = nn.ModuleList()
        in_channels = stem_channels
        for expansion_ratio, out_channels, num_blocks, stride, kernel_size, se_ratio in blocks_config:
            out_channels = round_channels(out_channels)
            num_blocks = int(num_blocks * depth_multiplier)
            
            for i in range(num_blocks):
                block_stride = stride if i == 0 else 1
                self.blocks.append(
                    MBConv(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        expansion_ratio=expansion_ratio,
                        se_ratio=se_ratio,
                    )
                )
                in_channels = out_channels

        # Head
        head_channels = round_channels(1280)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
        )

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate if dropout_rate > 0 else dropout),
            nn.Linear(head_channels, num_classes),
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
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        
        return x


