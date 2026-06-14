from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

from .text_descriptions import (
    DEFAULT_UNLABELED_TEXT,
    EP_FCHD_CLASS_DESCRIPTIONS,
    EP_FCHD_CLASS_NAMES,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def encode_texts(tokenizer, texts: list[str], max_length: int, device=None) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    return encoded


class EPFCHDImageTextDataset(Dataset):
    """Folder dataset for EP_FCHD.

    Expected labeled splits:
        split_dir/3VT_abnormal/*.png
        split_dir/3VT_Norm/*.png
        split_dir/A4C_abnormal/*.png
        split_dir/A4C_Norm/*.png

    Expected unlabeled split:
        split_dir/unlabeled/*.png
    """

    def __init__(
        self,
        split_dir: Path | str,
        tokenizer,
        image_processor,
        class_names: Iterable[str] = EP_FCHD_CLASS_NAMES,
        class_descriptions: dict[str, str] = EP_FCHD_CLASS_DESCRIPTIONS,
        max_length: int = 128,
        unlabeled: bool = False,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.class_names = list(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.class_descriptions = dict(class_descriptions)
        self.max_length = max_length
        self.unlabeled = unlabeled

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Dataset split does not exist: {self.split_dir}")

        self.samples: list[tuple[Path, int]] = []
        if unlabeled:
            roots = [self.split_dir]
            unlabeled_child = self.split_dir / "unlabeled"
            if unlabeled_child.exists():
                roots = [unlabeled_child]
            for root in roots:
                self.samples.extend((path, -1) for path in list_images(root))
        else:
            for class_name in self.class_names:
                class_dir = self.split_dir / class_name
                if not class_dir.exists():
                    raise FileNotFoundError(
                        f"Missing EP_FCHD class folder: {class_dir}. "
                        f"Expected folders: {', '.join(self.class_names)}"
                    )
                for path in list_images(class_dir):
                    self.samples.append((path, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.split_dir}")

        self.class_text_encodings: dict[str, dict[str, torch.Tensor]] = {}
        for class_name in self.class_names:
            if class_name not in self.class_descriptions:
                raise KeyError(f"Missing text description for class: {class_name}")
            text = self.class_descriptions[class_name]
            encoded = encode_texts(tokenizer, [text], max_length=max_length)
            self.class_text_encodings[class_name] = {
                key: value[0] for key, value in encoded.items()
            }
        default_encoded = encode_texts(tokenizer, [DEFAULT_UNLABELED_TEXT], max_length=max_length)
        self.default_text_encoding = {key: value[0] for key, value in default_encoded.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image_tensor = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

        if label >= 0:
            class_name = self.idx_to_class[label]
            text_encoding = self.class_text_encodings[class_name]
        else:
            class_name = "unlabeled"
            text_encoding = self.default_text_encoding

        return {
            "image": image_tensor,
            "text_encodings": {
                key: value.clone() for key, value in text_encoding.items()
            },
            "label": torch.tensor(label, dtype=torch.long),
            "class_name": class_name,
            "filename": str(path.relative_to(self.split_dir)),
        }

    def label_distribution(self) -> dict[str, int]:
        counts = {class_name: 0 for class_name in self.class_names}
        for _, label in self.samples:
            if label >= 0:
                counts[self.idx_to_class[label]] += 1
        return counts


def stratified_labeled_unlabeled_split(
    dataset: EPFCHDImageTextDataset,
    labeled_ratio: float,
    seed: int = 42,
    min_labeled_per_class: int = 1,
) -> tuple[Subset, Subset | None]:
    if not 0 < labeled_ratio <= 1:
        raise ValueError("labeled_ratio must be in (0, 1].")
    if labeled_ratio >= 1:
        return Subset(dataset, list(range(len(dataset)))), None

    rng = random.Random(seed)
    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    labeled_indices: list[int] = []
    unlabeled_indices: list[int] = []
    for label, indices in class_indices.items():
        rng.shuffle(indices)
        n_labeled = max(min_labeled_per_class, int(np.ceil(len(indices) * labeled_ratio)))
        n_labeled = min(n_labeled, len(indices))
        labeled_indices.extend(indices[:n_labeled])
        unlabeled_indices.extend(indices[n_labeled:])

    rng.shuffle(labeled_indices)
    rng.shuffle(unlabeled_indices)
    return Subset(dataset, labeled_indices), Subset(dataset, unlabeled_indices)
