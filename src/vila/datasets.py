from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


class AVACaptionPairDataset(Dataset):
    """Lightweight image/comment dataset scaffold for future VILA-lite adaptation."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path | None = None,
        image_transform=None,
        text_transform=None,
    ):
        if "image_path" not in frame.columns or "comment" not in frame.columns:
            raise ValueError("Manifest must contain at least image_path and comment columns.")
        self.frame = frame.reset_index(drop=True).copy()
        self.image_root = Path(image_root) if image_root else None
        self.image_transform = image_transform
        self.text_transform = text_transform

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        image_root: str | Path | None = None,
        image_transform=None,
        text_transform=None,
    ) -> "AVACaptionPairDataset":
        frame = pd.read_csv(csv_path)
        return cls(
            frame=frame,
            image_root=image_root,
            image_transform=image_transform,
            text_transform=text_transform,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        if self.image_root is not None:
            return (self.image_root / path).resolve(strict=False)
        return path.resolve(strict=False)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        resolved_path = self._resolve_image_path(str(row["image_path"]))
        with Image.open(resolved_path) as img:
            image = img.convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        text = "" if pd.isna(row["comment"]) else str(row["comment"])
        if self.text_transform is not None:
            text = self.text_transform(text)

        mos = None
        if "mos" in row.index and not pd.isna(row["mos"]):
            mos = float(row["mos"])

        return {
            "image": image,
            "text": text,
            "mos": mos,
            "image_path": str(resolved_path),
            "image_id": str(row["image_id"]) if "image_id" in row.index else None,
        }


def collate_vila_examples(examples: list[dict[str, object]]) -> dict[str, object]:
    mos_values = [example["mos"] for example in examples]
    return {
        "images": [example["image"] for example in examples],
        "texts": [example["text"] for example in examples],
        "mos": torch.tensor(
            [float(value) if value is not None else float("nan") for value in mos_values],
            dtype=torch.float32,
        ),
        "image_paths": [example["image_path"] for example in examples],
        "image_ids": [example["image_id"] for example in examples],
    }


def make_caption_dataloader(
    csv_path: str | Path,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    image_root: str | Path | None = None,
    image_transform=None,
    text_transform=None,
) -> DataLoader:
    dataset = AVACaptionPairDataset.from_csv(
        csv_path=csv_path,
        image_root=image_root,
        image_transform=image_transform,
        text_transform=text_transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_vila_examples,
    )
