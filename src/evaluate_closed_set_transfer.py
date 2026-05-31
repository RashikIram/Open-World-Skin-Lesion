from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import ExperimentConfig, KNOWN_CLASSES, SEED
from .datasets import DomainAdaptationDataset
from .metrics import evaluate_closed_set
from .models import build_model
from .preprocessing import add_known_unknown_columns, load_standardized_splits
from .transforms import get_eval_transform
from .utils import ensure_dir, get_device, load_model_state, seed_everything


def _jsonable(obj: Any):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    return obj


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2)


def label_name(label_id: int, class_names: Sequence[str]) -> str:
    label_id = int(label_id)
    if 0 <= label_id < len(class_names):
        return str(class_names[label_id])
    return f"label_{label_id}"


def save_closed_set_outputs(
    output_dir: Path,
    split_name: str,
    metrics: dict,
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
):
    split_dir = ensure_dir(output_dir / split_name)
    class_names = list(KNOWN_CLASSES)
    labels = list(range(len(class_names)))

    save_json(metrics, split_dir / "metrics.json")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        digits=4,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(split_dir / "classification_report.csv")
    save_json(report, split_dir / "classification_report.json")

    pred_df = df.reset_index(drop=True).copy()
    pred_df["eval_split"] = split_name
    pred_df["y_true"] = y_true.astype(int)
    pred_df["y_true_label"] = [label_name(x, class_names) for x in y_true]
    pred_df["y_pred"] = y_pred.astype(int)
    pred_df["y_pred_label"] = [label_name(x, class_names) for x in y_pred]

    for i, name in enumerate(class_names):
        pred_df[f"prob_{name}"] = y_prob[:, i]

    pred_df.to_csv(split_dir / "predictions.csv", index=False)

    cm_count = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)

    pd.DataFrame(cm_count, index=class_names, columns=class_names).to_csv(
        split_dir / "confusion_matrix_counts.csv"
    )
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(
        split_dir / "confusion_matrix_normalized.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    disp.plot(ax=ax, values_format=".4f", xticks_rotation=45, colorbar=True)
    ax.set_title(f"{split_name} normalized confusion matrix")
    fig.tight_layout()
    fig.savefig(split_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    y_bin = label_binarize(y_true, classes=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    rows = []

    for i, name in enumerate(class_names):
        if i >= y_bin.shape[1] or len(np.unique(y_bin[:, i])) < 2:
            continue
        fpr, tpr, thresholds = roc_curve(y_bin[:, i], y_prob[:, i])
        class_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} AUC={class_auc:.4f}")
        rows.extend(
            {
                "class": name,
                "fpr": float(fp),
                "tpr": float(tp),
                "threshold": float(th),
                "auc": float(class_auc),
            }
            for fp, tp, th in zip(fpr, tpr, thresholds)
        )

    if rows:
        pd.DataFrame(rows).to_csv(split_dir / "roc_curve_data.csv", index=False)
        ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
        ax.set_title(f"{split_name} ROC curves")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(split_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_target_split(
    *,
    model,
    df: pd.DataFrame,
    tokenizer,
    text_col: str,
    batch_size: int,
    device,
    criterion,
    cfg: ExperimentConfig,
    split_name: str,
    output_dir: Path,
):
    if df.empty:
        print(f"Skipping {split_name}: no known target rows.")
        return None

    dataset = DomainAdaptationDataset(
        df,
        tokenizer,
        get_eval_transform(),
        text_col,
        domain_label=None,
        max_text_len=cfg.max_text_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 0),
    )

    metrics, y_true, y_pred, y_prob = evaluate_closed_set(
        model,
        loader,
        criterion,
        device,
        len(KNOWN_CLASSES),
    )

    save_closed_set_outputs(
        output_dir=output_dir,
        split_name=split_name,
        metrics=metrics,
        df=df,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )
    return metrics


def run(args):
    seed_everything(SEED)

    cfg = ExperimentConfig()
    device = get_device()

    output_dir = ensure_dir(
        Path(args.output_dir)
        / args.model_family
        / args.text_col
        / args.target_dataset.replace("/", "_").replace(" ", "_")
        / "direct_closed_set_transfer"
    )

    image_model_name = (
        cfg.resnet_model_name
        if args.model_family.startswith("resnet50")
        else cfg.image_model_name
    )

    df = load_standardized_splits(
        args.standardized_csv,
        args.target_image_roots,
        args.target_dataset,
    )
    df = add_known_unknown_columns(df)

    known_df = df[~df["is_unknown"]].copy()
    val_df = known_df[known_df["split"] == "target_val"].reset_index(drop=True)
    test_df = known_df[known_df["split"] == "target_test"].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    model = build_model(
        args.model_family,
        image_model_name,
        cfg.text_model_name,
        len(KNOWN_CLASSES),
        cfg.fusion_dim,
        cfg.num_heads,
    ).to(device)

    missing, unexpected = load_model_state(model, args.checkpoint, strict=False)
    save_json(
        {
            "checkpoint": str(args.checkpoint),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        },
        output_dir / "loaded_checkpoint_info.json",
    )

    criterion = nn.CrossEntropyLoss()

    metrics = {
        "metadata": {
            "target_dataset": args.target_dataset,
            "model_family": args.model_family,
            "text_col": args.text_col,
            "checkpoint": str(args.checkpoint),
            "known_classes": list(KNOWN_CLASSES),
            "evaluation": "direct_closed_set_transfer_on_known_target_classes",
        },
        "split_sizes": {
            "target_val_known": int(len(val_df)),
            "target_test_known": int(len(test_df)),
        },
    }

    val_metrics = evaluate_target_split(
        model=model,
        df=val_df,
        tokenizer=tokenizer,
        text_col=args.text_col,
        batch_size=args.batch_size,
        device=device,
        criterion=criterion,
        cfg=cfg,
        split_name="target_val_known",
        output_dir=output_dir,
    )
    if val_metrics is not None:
        metrics["target_val_known"] = val_metrics

    test_metrics = evaluate_target_split(
        model=model,
        df=test_df,
        tokenizer=tokenizer,
        text_col=args.text_col,
        batch_size=args.batch_size,
        device=device,
        criterion=criterion,
        cfg=cfg,
        split_name="target_test_known",
        output_dir=output_dir,
    )
    if test_metrics is not None:
        metrics["target_test_known"] = test_metrics

    save_json(metrics, output_dir / "direct_closed_set_transfer_metrics.json")

    rows = []
    for split in ["target_val_known", "target_test_known"]:
        if split in metrics:
            rows.append({"split": split, **metrics[split]})
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "direct_closed_set_transfer_summary.csv", index=False)

    print("Saved direct closed-set transfer outputs to", output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardized-csv", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--target-image-roots", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-family", choices=["mobilevit_cross_attention", "mobilevit_gated", "mobilevit_concat", "resnet50_cross_attention", "resnet50_gated", "resnet50_concat"], default="mobilevit_gated")
    parser.add_argument(
        "--text-col",
        choices=["text_core", "text_full", "text_missing_explicit"],
        default="text_full",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
