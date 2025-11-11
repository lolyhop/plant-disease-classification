import torch
from torch import nn
from torchvision import models


class ResNet(nn.Module):
    """ResNet model for plant disease classification."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if model_name == "resnet18":
            backbone = models.resnet18(weights="DEFAULT" if pretrained else None)
        elif model_name == "resnet34":
            backbone = models.resnet34(weights="DEFAULT" if pretrained else None)
        elif model_name == "resnet50":
            backbone = models.resnet50(weights="DEFAULT" if pretrained else None)
        elif model_name == "resnet101":
            backbone = models.resnet101(weights="DEFAULT" if pretrained else None)
        else:
            raise ValueError(f"Unknown ResNet model: {model_name}")

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone.fc.in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)


class DenseNet(nn.Module):
    """DenseNet model for plant disease classification."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "densenet121",
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if model_name == "densenet121":
            backbone = models.densenet121(weights="DEFAULT" if pretrained else None)
        elif model_name == "densenet169":
            backbone = models.densenet169(weights="DEFAULT" if pretrained else None)
        elif model_name == "densenet201":
            backbone = models.densenet201(weights="DEFAULT" if pretrained else None)
        else:
            raise ValueError(f"Unknown DenseNet model: {model_name}")

        self.features = backbone.features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone.classifier.in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = torch.relu(features, inplace=True)
        out = torch.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)


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

