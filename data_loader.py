"""Data loading and preprocessing utilities for LFW face recognition experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


@dataclass
class SplitData:
    """Container for split image arrays and encoded labels."""

    train_images: List[np.ndarray]
    val_images: List[np.ndarray]
    test_images: List[np.ndarray]
    train_labels: np.ndarray
    val_labels: np.ndarray
    test_labels: np.ndarray
    label_encoder: LabelEncoder


def set_seed(seed: int = 42) -> None:
    """Set RNG seeds for reproducible train/val/test splits."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def pil_to_bgr(pil_img, size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    """Convert PIL image to resized OpenCV BGR image."""
    img = pil_img.convert("RGB").resize(size)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def load_lfw_arrays(
    dataset_name: str = "bitmind/lfw",
    split: str = "train",
    size: Tuple[int, int] = (160, 160),
) -> Tuple[List[np.ndarray], List[str], DatasetDict]:
    """Load LFW from Hugging Face and return BGR images and string labels."""
    ds = load_dataset(dataset_name)
    records = ds[split]
    images = [pil_to_bgr(item["image"], size=size) for item in records]
    labels = [str(item["label"]) for item in records]
    return images, labels, ds


def filter_identities(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    min_images_per_id: int = 5,
) -> Tuple[List[np.ndarray], List[str]]:
    """Keep identities with at least min_images_per_id samples."""
    counts = Counter(labels)
    valid_ids = {label for label, count in counts.items() if count >= min_images_per_id}
    filtered = [(img, lbl) for img, lbl in zip(images, labels) if lbl in valid_ids]
    f_images, f_labels = zip(*filtered)
    return list(f_images), list(f_labels)


def stratified_split(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> SplitData:
    """Create stratified 80/10/10-style split and a shared label encoder."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    idx = np.arange(len(images))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=(1.0 - train_ratio),
        stratify=labels,
        random_state=seed,
    )

    val_share = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_share),
        stratify=np.array(labels)[temp_idx],
        random_state=seed,
    )

    le = LabelEncoder()
    le.fit(labels)

    train_labels = le.transform(np.array(labels)[train_idx])
    val_labels = le.transform(np.array(labels)[val_idx])
    test_labels = le.transform(np.array(labels)[test_idx])

    return SplitData(
        train_images=[images[i] for i in train_idx],
        val_images=[images[i] for i in val_idx],
        test_images=[images[i] for i in test_idx],
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        label_encoder=le,
    )


class FaceDataset(Dataset):
    """Torch dataset returning transformed RGB tensors and integer labels."""

    def __init__(
        self,
        images: Sequence[np.ndarray],
        labels: Sequence[int],
        transform: Optional[Callable] = None,
    ) -> None:
        self.images = list(images)
        self.labels = np.array(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = cv2.cvtColor(self.images[idx], cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def cache_numpy_arrays(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    cache_dir: str | Path = "lfw_cache",
    prefix: str = "lfw",
) -> Dict[str, Path]:
    """Persist images and labels as NumPy arrays for faster subsequent loads."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    img_file = cache_path / f"{prefix}_images.npy"
    lbl_file = cache_path / f"{prefix}_labels.npy"

    np.save(img_file, np.array(images, dtype=np.uint8), allow_pickle=False)
    np.save(lbl_file, np.array(labels), allow_pickle=False)

    return {"images": img_file, "labels": lbl_file}
