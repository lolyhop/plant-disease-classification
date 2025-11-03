import typing as tp
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class EarlyStopping:
    def __init__(self, patience: int, delta: float = 0.0) -> None:
        self.patience = patience
        self.delta = delta
        self.best_score: tp.Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> None:
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class TensorboardLogger:
    def __init__(self, log_dir: Path) -> None:
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self.writer.close()
