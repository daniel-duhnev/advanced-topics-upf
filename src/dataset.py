import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch
from .config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, SEED, NUM_CLASSES
)


def get_transforms(split="train"):
    """Return augmentation pipeline for train/val/test."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])



def split_by_lesion_id(metadata_df):
    """Split dataset ensuring no lesion appears in multiple splits (prevents data leakage)."""
    gss_train = GroupShuffleSplit(n_splits=1, test_size=1 - TRAIN_RATIO, random_state=SEED)
    train_idx, temp_idx = next(gss_train.split(metadata_df, groups=metadata_df["lesion_id"]))

    temp_df = metadata_df.iloc[temp_idx]
    val_fraction = VAL_RATIO / (VAL_RATIO + (1 - TRAIN_RATIO - VAL_RATIO))
    gss_val = GroupShuffleSplit(n_splits=1, test_size=1 - val_fraction, random_state=SEED)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["lesion_id"]))

    train_df = metadata_df.iloc[train_idx]
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    return train_df, val_df, test_df


def compute_class_weights(train_df):
    """Compute inverse frequency class weights for handling imbalance."""
    class_counts = train_df["dx"].value_counts().sort_index()
    total = len(train_df)
    weights = total / (NUM_CLASSES * class_counts.values)
    return weights.astype(np.float32)


def resolve_image_dir(data_dir):
    """Find where images actually live — handles Kaggle's part_1/part_2 structure."""
    data_dir = Path(data_dir)

    # Check if flat images/ directory exists
    if (data_dir / "images").is_dir():
        return data_dir / "images"

    # Check if Kaggle-style part_1/part_2 exist
    part1 = data_dir / "HAM10000_images_part_1"
    part2 = data_dir / "HAM10000_images_part_2"
    if part1.is_dir() or part2.is_dir():
        # Create unified images/ dir with symlinks or just search both
        return [p for p in [part1, part2] if p.is_dir()]

    # Fallback: images might be directly in data_dir
    if any(data_dir.glob("*.jpg")):
        return data_dir

    raise FileNotFoundError(
        f"Cannot find images in {data_dir}. Expected 'images/', "
        f"'HAM10000_images_part_1/', or .jpg files directly in the directory."
    )


class HAM10000Dataset(Dataset):
    """PyTorch Dataset for the HAM10000 skin lesion dataset."""

    def __init__(self, df, image_dirs, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dirs = [Path(d) for d in image_dirs] if isinstance(image_dirs, list) else [Path(image_dirs)]
        self.transform = transform
        # Fixed label map — must be consistent across all splits
        self.label_map = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}

    def __len__(self):
        return len(self.df)

    def _find_image(self, image_id):
        for d in self.image_dirs:
            path = d / f"{image_id}.jpg"
            if path.exists():
                return path
        raise FileNotFoundError(f"Image {image_id}.jpg not found in {self.image_dirs}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._find_image(row["image_id"])
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row["dx"]]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(metadata_path, image_dir):
    """Create train/val/test DataLoaders with proper splitting and augmentation."""
    metadata = pd.read_csv(metadata_path)
    train_df, val_df, test_df = split_by_lesion_id(metadata)

    image_dirs = resolve_image_dir(image_dir) if not isinstance(image_dir, list) else image_dir

    train_dataset = HAM10000Dataset(train_df, image_dirs, transform=get_transforms("train"))
    val_dataset = HAM10000Dataset(val_df, image_dirs, transform=get_transforms("val"))
    test_dataset = HAM10000Dataset(test_df, image_dirs, transform=get_transforms("test"))

    num_workers = 0 if torch.backends.mps.is_available() else 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    class_weights = compute_class_weights(train_df)

    return train_loader, val_loader, test_loader, class_weights
