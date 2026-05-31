from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


DEFAULT_SAMPLER_SEED = 42


def _validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _safe_text_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


class SkinLesionTextImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        transform,
        text_col: str,
        label_col: str = "label_id",
        domain_label: Optional[int] = None,
        max_text_len: int = 96,
    ):
        required_columns = ["image_path", label_col]

        # The notebook's image-only runs intentionally use an empty text string.
        # For every real text experiment, fail early if preprocessing did not
        # create the requested text column.
        if text_col != "image_only":
            required_columns.append(text_col)

        _validate_dataframe_columns(
            df=df,
            required_columns=required_columns,
            dataset_name=self.__class__.__name__,
        )

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

        if self.text_col == "image_only":
            text = ""
        else:
            text = _safe_text_value(row[self.text_col])

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

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
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        transform,
        text_col: str,
        max_text_len: int = 96,
    ):
        super().__init__(
            df=df,
            tokenizer=tokenizer,
            transform=transform,
            text_col=text_col,
            label_col="label_open_id",
            domain_label=None,
            max_text_len=max_text_len,
        )


class ImageOnlyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform,
        label_col: str = "label_id",
        domain_label: Optional[int] = None,
    ):
        _validate_dataframe_columns(
            df=df,
            required_columns=["image_path", label_col],
            dataset_name=self.__class__.__name__,
        )

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


def make_weighted_random_sampler(
    df: pd.DataFrame,
    label_col: str = "label_id",
    num_classes: Optional[int] = None,
    seed: int = DEFAULT_SAMPLER_SEED,
) -> WeightedRandomSampler:
    """
    Notebook-equivalent sampler.

    This matches the closed-set notebook behaviour:
    - inverse class-frequency weights
    - one sampled item per training row
    - replacement=True
    - deterministic torch.Generator seed
    """

    _validate_dataframe_columns(
        df=df,
        required_columns=[label_col],
        dataset_name="make_weighted_random_sampler input dataframe",
    )

    labels = df[label_col].to_numpy(dtype=int)

    if len(labels) == 0:
        raise ValueError("Cannot build a WeightedRandomSampler from an empty dataframe.")

    if num_classes is None:
        num_classes = int(labels.max()) + 1

    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)

    sample_weights = class_weights[labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def effective_number_weights(labels, beta: float = 0.99) -> dict[int, float]:
    labels = np.asarray(labels, dtype=int)
    classes, counts = np.unique(labels, return_counts=True)

    weights = {}
    for cls, count in zip(classes, counts):
        effective_num = 1.0 - np.power(beta, count)
        weights[int(cls)] = float((1.0 - beta) / max(effective_num, 1e-12))

    mean_weight = np.mean(list(weights.values())) if weights else 1.0
    return {k: v / mean_weight for k, v in weights.items()}


def make_soft_weighted_random_sampler(
    df: pd.DataFrame,
    beta: float = 0.99,
    label_col: str = "label_id",
    seed: int = DEFAULT_SAMPLER_SEED,
) -> WeightedRandomSampler:
    """
    Effective-number weighted sampler.

    Kept for the improved modular pipeline, but now also seeded so repeated
    experiments are deterministic like the original notebook sampler.
    """

    _validate_dataframe_columns(
        df=df,
        required_columns=[label_col],
        dataset_name="make_soft_weighted_random_sampler input dataframe",
    )

    labels = df[label_col].to_numpy(dtype=int)

    if len(labels) == 0:
        raise ValueError("Cannot build a WeightedRandomSampler from an empty dataframe.")

    weights_by_class = effective_number_weights(labels, beta=beta)
    sample_weights = torch.DoubleTensor(
        [weights_by_class[int(y)] for y in labels]
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def get_loss_weights_from_train_df(
    df: pd.DataFrame,
    num_classes: int,
    beta: float = 0.99,
    label_col: str = "label_id",
) -> torch.Tensor:
    _validate_dataframe_columns(
        df=df,
        required_columns=[label_col],
        dataset_name="get_loss_weights_from_train_df input dataframe",
    )

    weights_by_class = effective_number_weights(
        df[label_col].values,
        beta=beta,
    )

    weights = [weights_by_class.get(i, 1.0) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)
