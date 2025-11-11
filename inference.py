import argparse
import typing as tp
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data_utils.dataset import PlantDataset
from src.data_utils.utils import worker_init_fn
from src.models import cnn, mlp, t2t_vit, vit
from src.train_utils.logger import setup_logger
from src.train_utils.metrics import Metrics

MODEL_REGISTRY: tp.Dict[str, tp.Any] = {
    "vit_2021_orig": vit.VisionTransformer,
    "t2t_vit": t2t_vit.T2TViT,
    "mlp": mlp.MLP,
    "deep_mlp": mlp.DeepMLP,
    "resnet": cnn.ResNet,
    "densenet": cnn.DenseNet,
    "efficientnet": cnn.EfficientNet,
}


def inference(cfg: tp.Dict[str, tp.Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = Path(cfg["training"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = log_dir / "predictions.csv"
    metrics_path = log_dir / "metrics.csv"

    logger = setup_logger(cfg["training"].get("log_to_file", False), log_dir)
    logger.info("Starting inference...")

    test_dataset = PlantDataset(
        root=Path(cfg["data"]["root_dir"]) / "valid",
        image_size=cfg["data"]["image_size"],
        is_train=False,
    )
    logger.info(f"Test dataset ready: {len(test_dataset)} samples")

    # Create dataloader for inference
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    logger.info(f"Test loader ready: {len(test_loader)} batches")

    # Initialize model
    ModelClass = MODEL_REGISTRY[cfg["model"]["name"]]
    model = ModelClass(**cfg["model"]["params"]).to(device)

    # Load weights
    weights_path = log_dir / "best_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    logger.info(f"Loaded model weights from {weights_path}")

    # Prepare metrics
    metrics = Metrics(num_classes=cfg["data"]["num_classes"])
    all_predictions = []
    all_paths = []
    all_labels = []

    # Inference loop
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            metrics.update(logits, labels)

            all_predictions.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            batch_indices = range(len(all_paths), len(all_paths) + len(images))
            for idx in batch_indices:
                path, _ = test_dataset.samples[idx]
                all_paths.append(str(path))

    # Compute metrics
    results = metrics.compute()
    logger.info("Inference metrics:")
    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}")

    # Save predictions.csv
    with open(predictions_path, "w") as f:
        f.write("index,image_path,true_label,prediction\n")
        for i, (path, true_label, pred) in enumerate(
            zip(all_paths, all_labels, all_predictions)
        ):
            f.write(f"{i},{path},{true_label},{pred}\n")

    # Save metrics.csv
    with open(metrics_path, "w") as f:
        f.write("metric,value\n")
        for k, v in results.items():
            f.write(f"{k},{v:.6f}\n")

    logger.info(f"Predictions saved to {predictions_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info("Inference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    inference(cfg)
