import typing as tp

import torch
import torchmetrics


class Metrics:
    def __init__(self, num_classes: int) -> None:
        self.num_classes: int = num_classes
        device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Classification metrics
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        ).to(device)
        self.precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device)
        self.recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device)
        self.f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device)

    def reset(self) -> None:
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.accuracy.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        self.f1.update(preds, targets)

    def compute(self) -> tp.Dict[str, float]:
        return {
            "accuracy": self.accuracy.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
            "f1": self.f1.compute().item(),
        }
