from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


class SkinLesionTextImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, transform, text_col: str, label_col: str = "label_id", domain_label: Optional[int] = None, max_text_len: int = 96):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.text_col = text_col
        self.label_col = label_col
        self.domain_label = domain_label
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        pixel_values = self.transform(image)
        text = "" if self.text_col == "image_only" else str(row.get(self.text_col, ""))
        text_inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_text_len, return_tensors="pt")
        item = {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row[self.label_col]), dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }
        if self.domain_label is not None:
            item["domain"] = torch.tensor(self.domain_label, dtype=torch.long)
        return item


class PadUfesClosedSetDataset(SkinLesionTextImageDataset):
    pass


class DomainAdaptationDataset(SkinLesionTextImageDataset):
    pass


class OpenWorldDataset(SkinLesionTextImageDataset):
    def __init__(self, df, tokenizer, transform, text_col: str, max_text_len: int = 96):
        super().__init__(df, tokenizer, transform, text_col, label_col="label_open_id", domain_label=None, max_text_len=max_text_len)


class ImageOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, label_col: str = "label_id", domain_label: Optional[int] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col
        self.domain_label = domain_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        item = {
            "pixel_values": self.transform(image),
            "label": torch.tensor(int(row[self.label_col]), dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }
        if self.domain_label is not None:
            item["domain"] = torch.tensor(self.domain_label, dtype=torch.long)
        return item


def effective_number_weights(labels, beta: float = 0.99) -> dict[int, float]:
    labels = np.asarray(labels, dtype=int)
    classes, counts = np.unique(labels, return_counts=True)
    weights = {}
    for cls, count in zip(classes, counts):
        effective_num = 1.0 - np.power(beta, count)
        weights[int(cls)] = float((1.0 - beta) / max(effective_num, 1e-12))
    mean_weight = np.mean(list(weights.values())) if weights else 1.0
    return {k: v / mean_weight for k, v in weights.items()}


def make_soft_weighted_random_sampler(df: pd.DataFrame, beta: float = 0.99, label_col: str = "label_id") -> WeightedRandomSampler:
    weights_by_class = effective_number_weights(df[label_col].values, beta=beta)
    sample_weights = [weights_by_class[int(y)] for y in df[label_col].values]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def get_loss_weights_from_train_df(df: pd.DataFrame, num_classes: int, beta: float = 0.99, label_col: str = "label_id") -> torch.Tensor:
    weights_by_class = effective_number_weights(df[label_col].values, beta=beta)
    weights = [weights_by_class.get(i, 1.0) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)
