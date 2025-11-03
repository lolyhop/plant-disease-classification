import typing as tp

import torch
import torchmetrics


class Metrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

        # Classification metrics
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

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
