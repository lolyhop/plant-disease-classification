import argparse
import typing as tp
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data_utils.dataset import PlantDataset
from src.data_utils.utils import worker_init_fn
from src.models import cnn, mlp, t2t_vit, vit
from src.train_utils.callbacks import EarlyStopping, TensorboardLogger
from src.train_utils.logger import setup_logger
from src.train_utils.metrics import Metrics
from src.train_utils.optim import build_optimizer, clip_gradients, compute_grad_norm

MODEL_REGISTRY: tp.Dict[str, tp.Any] = {
    "vit_2021_orig": vit.VisionTransformer,
    "t2t_vit": t2t_vit.T2TViT,
    "mlp": mlp.MLP,
    "deep_mlp": mlp.DeepMLP,
    "resnet": cnn.ResNet,
    "densenet": cnn.DenseNet,
    "efficientnet": cnn.EfficientNet,
}


def train(cfg: tp.Dict[str, tp.Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = Path(cfg["training"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(cfg["training"].get("log_to_file", False), log_dir)
    tb_logger = TensorboardLogger(log_dir / "tensorboard")

    train_dataset = PlantDataset(
        root=Path(cfg["data"]["root_dir"]) / "train",
        image_size=cfg["data"]["image_size"],
        augmentations=cfg.get("augmentations"),
        is_train=True,
    )
    val_dataset = PlantDataset(
        root=Path(cfg["data"]["root_dir"]) / "valid",
        image_size=cfg["data"]["image_size"],
        is_train=False,
    )
    logger.info(f"Datasets ready: train={len(train_dataset)}, val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    logger.info(
        f"Dataloaders ready: train_loader={len(train_loader)} batches, val_loader={len(val_loader)} batches"
    )

    ModelClass = MODEL_REGISTRY[cfg["model"]["name"]]
    model = ModelClass(**cfg["model"]["params"]).to(device)
    logger.info(f"Model instantiated: {cfg['model']['name']}")

    optimizer = build_optimizer(
        model,
        name=cfg["training"]["optimizer"],
        lr=float(cfg["training"]["lr"]),
        **cfg["training"].get("optimizer_kwargs", {}),
    )

    criterion = nn.CrossEntropyLoss()
    early_stopper = (
        EarlyStopping(patience=cfg["training"]["patience"])
        if "patience" in cfg["training"]
        else None
    )
    metrics = Metrics(num_classes=cfg["data"]["num_classes"])

    num_epochs = cfg["training"]["num_epochs"]
    max_grad_norm = cfg["training"].get("clip_grad_norm")
    best_val_acc = 0.0
    logger.info("Training setup complete. Starting training loop...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_grad_norm = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            grad_norm = compute_grad_norm(model)
            total_grad_norm += grad_norm

            if max_grad_norm:
                clip_gradients(model, max_grad_norm)

            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_grad_norm = total_grad_norm / len(train_loader)

        logger.info(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"TrainLoss={avg_train_loss:.4f} | GradNorm={avg_grad_norm:.2f}"
        )

        tb_logger.log_scalar("train/loss", avg_train_loss, epoch)
        tb_logger.log_scalar("train/grad_norm", avg_grad_norm, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        metrics.reset()

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item()
                metrics.update(logits, labels)

        avg_val_loss = val_loss / len(val_loader)
        val_results = metrics.compute()
        val_acc = val_results["accuracy"]

        tb_logger.log_scalar("val/loss", avg_val_loss, epoch)
        for key, value in val_results.items():
            tb_logger.log_scalar(f"val/{key}", value, epoch)

        logger.info(
            f"Validation: Loss={avg_val_loss:.4f} | "
            f"Acc={val_acc:.4f} | F1={val_results['f1']:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), log_dir / "best_model.pt")

        if early_stopper:
            early_stopper.step(val_acc)
            if early_stopper.should_stop:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    tb_logger.close()
    logger.info(f"Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)
