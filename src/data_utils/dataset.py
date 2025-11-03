import typing as tp
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


_AUGMENTATION_MAP: tp.Dict[str, tp.Any] = {
    "horizontal_flip": transforms.RandomHorizontalFlip,
    "vertical_flip": transforms.RandomVerticalFlip,
    "rotation": transforms.RandomRotation,
    "color_jitter": transforms.ColorJitter,
    "grayscale": transforms.RandomGrayscale,
}


class PlantDataset(Dataset):
    """Custom dataset for plant disease classification."""

    def __init__(
        self,
        root: tp.Union[str, Path],
        image_size: int = 224,
        augmentations: tp.Optional[tp.List[tp.Dict[str, tp.Any]]] = None,
        is_train: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.is_train = is_train
        self.samples: tp.List[tp.Tuple[Path, int]] = []
        self.class_to_idx: tp.Dict[str, int] = {}
        self._collect_samples()

        self.transform = self._build_transforms(augmentations or [])

    def _collect_samples(self) -> None:
        classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        for cls in classes:
            for img_path in (self.root / cls).glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[cls]))

    def _build_transforms(self, aug_cfg: tp.List[tp.Dict[str, tp.Any]]) -> transforms.Compose:
        t_list: tp.List[tp.Any] = []

        if self.is_train:
            for aug in aug_cfg:
                name = aug["name"]
                params = aug.get("params", {})
                if name not in _AUGMENTATION_MAP:
                    raise ValueError(f"Unknown augmentation: {name}, available augmentations are: {_AUGMENTATION_MAP.keys()}")
                t_list.append(_AUGMENTATION_MAP[name](**params))

        t_list.extend([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return transforms.Compose(t_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tp.Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def build_dataloaders(
    data_root: tp.Union[str, Path],
    image_size: int,
    batch_size: int,
    num_workers: int,
    augmentations: tp.Optional[tp.List[tp.Dict[str, tp.Any]]] = None,
) -> tp.Dict[str, DataLoader]:
    splits = ["train", "valid", "test"]
    data_root = Path(data_root)
    loaders: tp.Dict[str, DataLoader] = {}

    for split in splits:
        split_dir = data_root / split
        dataset = PlantDataset(
            root=split_dir,
            image_size=image_size,
            augmentations=augmentations if split == "train" else [],
            is_train=(split == "train"),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def get_num_classes(data_root: tp.Union[str, Path]) -> int:
    train_dir = Path(data_root) / "train"
    return len([p for p in train_dir.iterdir() if p.is_dir()])


__all__ = ["PlantDataset", "build_dataloaders", "get_num_classes"]
