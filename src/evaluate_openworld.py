from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import ExperimentConfig, KNOWN_CLASSES, UNKNOWN_LABEL_NAME, SEED
from .datasets import OpenWorldDataset
from .models import build_dann_model, build_model
from .preprocessing import add_known_unknown_columns, load_standardized_splits
from .transforms import get_eval_transform
from .utils import ensure_dir, get_device, load_model_state, seed_everything


# ============================================================
# JSON / plotting helpers
# ============================================================


def _json_clean(obj):
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    return obj


def save_json_clean(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(obj), f, indent=2)


def plot_confusion_matrix_png(y_true, y_pred, class_names: Sequence[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_title("Open-world normalized confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_unknown_roc_png(roc_df: pd.DataFrame, output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if roc_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(roc_df["fpr"], roc_df["tpr"], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Markdown-style open-world scoring
# ============================================================


def max_softmax_unknown_score(probs: np.ndarray) -> np.ndarray:
    """Max-softmax unknownness score kept for ablations only."""
    probs = np.asarray(probs)
    return 1.0 - np.max(probs, axis=1)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Energy-based unknownness score.

    The markdown open-world evaluation is energy-based:
    energy = -T * logsumexp(logits / T)

    Higher energy values are treated as more unknown, so samples are predicted
    UNKNOWN when energy >= the validation-selected threshold.
    """
    logits = np.asarray(logits, dtype=float)
    scaled = logits / temperature
    max_logits = np.max(scaled, axis=1, keepdims=True)
    logsumexp = max_logits.squeeze(1) + np.log(
        np.exp(scaled - max_logits).sum(axis=1)
    )
    return -temperature * logsumexp


def predict_open_world_from_threshold(
    known_pred: np.ndarray,
    unknown_score: np.ndarray,
    threshold: float,
    unknown_id: int,
) -> np.ndarray:
    pred = np.asarray(known_pred).copy()
    pred[np.asarray(unknown_score) >= threshold] = unknown_id
    return pred.astype(int)


def compute_oscr(
    is_known: np.ndarray,
    y_known_true: np.ndarray,
    known_pred: np.ndarray,
    known_confidence: np.ndarray,
) -> float:
    """Compute an OSCR-style area under CCR-vs-FPR curve."""

    is_known = np.asarray(is_known).astype(bool)
    y_known_true = np.asarray(y_known_true)
    known_pred = np.asarray(known_pred)
    known_confidence = np.asarray(known_confidence)

    thresholds = np.sort(np.unique(known_confidence))[::-1]
    if len(thresholds) < 2:
        return float("nan")

    n_known = max(int(is_known.sum()), 1)
    n_unknown = max(int((~is_known).sum()), 1)

    ccr = []
    fpr = []

    for threshold in thresholds:
        accepted = known_confidence >= threshold
        correct_known = accepted & is_known & (known_pred == y_known_true)
        false_known = accepted & (~is_known)
        ccr.append(correct_known.sum() / n_known)
        fpr.append(false_known.sum() / n_unknown)

    order = np.argsort(fpr)
    return float(np.trapz(np.asarray(ccr)[order], np.asarray(fpr)[order]))


def compute_open_world_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unknown_score: np.ndarray,
    unknown_id: int,
    known_pred: np.ndarray | None = None,
    known_confidence: np.ndarray | None = None,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    unknown_score = np.asarray(unknown_score)

    true_unknown = (y_true == unknown_id).astype(int)
    pred_unknown = (y_pred == unknown_id).astype(int)

    metrics = {
        "open_accuracy": float(accuracy_score(y_true, y_pred)),
        "open_macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "open_macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "open_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "open_weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "open_weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "open_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "unknown_precision": float(precision_score(true_unknown, pred_unknown, zero_division=0)),
        "unknown_recall": float(recall_score(true_unknown, pred_unknown, zero_division=0)),
        "unknown_f1": float(f1_score(true_unknown, pred_unknown, zero_division=0)),
    }

    if len(np.unique(true_unknown)) == 2:
        metrics["unknown_auroc"] = float(roc_auc_score(true_unknown, unknown_score))
    else:
        metrics["unknown_auroc"] = float("nan")

    if known_pred is not None and known_confidence is not None:
        metrics["oscr"] = compute_oscr(
            is_known=(y_true != unknown_id),
            y_known_true=np.where(y_true == unknown_id, -1, y_true),
            known_pred=known_pred,
            known_confidence=known_confidence,
        )
    else:
        metrics["oscr"] = float("nan")

    return metrics


def threshold_candidates(scores: np.ndarray, grid_size: int = 400) -> np.ndarray:
    """Candidate thresholds for energy-based unknown detection.

    Energy scores are not probabilities, so thresholds must not be clamped
    to [0, 1]. Search across the observed validation energy range, with a
    small margin so the grid can represent nearly-all-known and
    nearly-all-unknown decisions.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    if scores.size == 0:
        return np.asarray([0.0], dtype=float)

    lo = float(scores.min())
    hi = float(scores.max())

    if np.isclose(lo, hi):
        eps = max(abs(lo) * 1e-6, 1e-6)
        return np.asarray([lo - eps, lo, lo + eps], dtype=float)

    margin = 0.01 * (hi - lo)
    unique_scores = np.unique(scores)
    grid = np.linspace(lo - margin, hi + margin, grid_size)
    return np.unique(np.concatenate([unique_scores, grid]))


def find_best_unknown_threshold(
    y_open_true: np.ndarray,
    known_pred: np.ndarray,
    unknown_score: np.ndarray,
    unknown_id: int,
    grid_size: int = 400,
) -> dict:
    best = {
        "threshold": 0.5,
        "known_confidence_threshold": 0.5,
        "open_macro_f1": -1.0,
    }

    for threshold in threshold_candidates(unknown_score, grid_size=grid_size):
        pred_open = predict_open_world_from_threshold(
            known_pred=known_pred,
            unknown_score=unknown_score,
            threshold=float(threshold),
            unknown_id=unknown_id,
        )
        macro_f1 = float(f1_score(y_open_true, pred_open, average="macro", zero_division=0))
        if macro_f1 > best["open_macro_f1"]:
            best = {
                "threshold": float(threshold),
                "energy_threshold": float(threshold),
                "open_macro_f1": macro_f1,
            }

    return best


def unknown_roc_curve_frame(y_open_true, unknown_score, unknown_id: int) -> pd.DataFrame:
    y_unknown = (np.asarray(y_open_true).astype(int) == unknown_id).astype(int)
    if len(np.unique(y_unknown)) < 2:
        return pd.DataFrame(columns=["fpr", "tpr", "threshold"])
    fpr, tpr, thresholds = roc_curve(y_unknown, unknown_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


# ============================================================
# Model output collection and reporting
# ============================================================


@torch.no_grad()
def collect_outputs(model, loader, device):
    model.eval()
    logits_all, labels_all, indices_all = [], [], []

    for batch in tqdm(loader, desc="collect", leave=False):
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        out = model(batch["pixel_values"], batch.get("input_ids"), batch.get("attention_mask"))
        logits = out[0] if isinstance(out, tuple) else out
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.extend(batch["label"].detach().cpu().numpy().tolist())
        indices_all.extend(batch["index"].detach().cpu().numpy().tolist())

    if not logits_all:
        raise ValueError("No batches were produced. Check that the evaluated split is not empty.")

    return (
        np.concatenate(logits_all, axis=0),
        np.asarray(labels_all, dtype=int),
        np.asarray(indices_all, dtype=int),
    )


def build_prediction_dataframe(
    source_df: pd.DataFrame,
    indices: np.ndarray,
    logits: np.ndarray,
    probs: np.ndarray,
    y_open_true: np.ndarray,
    y_open_pred: np.ndarray,
    known_pred: np.ndarray,
    known_confidence: np.ndarray,
    unknown_score: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
    text_col: str,
    split_name: str,
) -> pd.DataFrame:
    pred_df = source_df.iloc[indices].copy().reset_index(drop=True)

    id_to_label = {i: name for i, name in enumerate(class_names)}
    known_id_to_label = {i: name for i, name in enumerate(KNOWN_CLASSES)}

    pred_df["eval_split"] = split_name
    pred_df["eval_text_col"] = text_col
    pred_df["threshold_unknown_score"] = float(threshold)
    pred_df["threshold_energy"] = float(threshold)

    pred_df["y_open_true"] = y_open_true.astype(int)
    pred_df["y_open_true_label"] = [id_to_label.get(int(x), str(x)) for x in y_open_true]
    pred_df["known_pred"] = known_pred.astype(int)
    pred_df["known_pred_label"] = [known_id_to_label.get(int(x), str(x)) for x in known_pred]
    pred_df["known_confidence"] = known_confidence
    pred_df["max_softmax_confidence"] = probs.max(axis=1)
    pred_df["energy_score"] = unknown_score
    pred_df["unknown_score"] = unknown_score
    pred_df["y_open_pred"] = y_open_pred.astype(int)
    pred_df["y_open_pred_label"] = [id_to_label.get(int(x), str(x)) for x in y_open_pred]

    for class_idx, class_name in enumerate(KNOWN_CLASSES):
        pred_df[f"logit_{class_name}"] = logits[:, class_idx]
        pred_df[f"prob_{class_name}"] = probs[:, class_idx]

    return pred_df


def save_split_outputs(
    split_name: str,
    output_dir: Path,
    pred_df: pd.DataFrame,
    metrics: dict,
    class_names: Sequence[str],
    unknown_id: int,
) -> None:
    labels = list(range(len(class_names)))

    pred_df.to_csv(output_dir / f"{split_name}_openworld_predictions.csv", index=False)

    report = classification_report(
        pred_df["y_open_true"],
        pred_df["y_open_pred"],
        labels=labels,
        target_names=list(class_names),
        zero_division=0,
        digits=4,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / f"{split_name}_openworld_classification_report.csv"
    )
    save_json_clean(report, output_dir / f"{split_name}_openworld_classification_report.json")

    cm = confusion_matrix(
        pred_df["y_open_true"],
        pred_df["y_open_pred"],
        labels=labels,
    )
    cm_norm = confusion_matrix(
        pred_df["y_open_true"],
        pred_df["y_open_pred"],
        labels=labels,
        normalize="true",
    )
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        output_dir / f"{split_name}_openworld_confusion_matrix_counts.csv"
    )
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(
        output_dir / f"{split_name}_openworld_confusion_matrix_normalized.csv"
    )
    plot_confusion_matrix_png(
        pred_df["y_open_true"],
        pred_df["y_open_pred"],
        class_names,
        output_dir / f"{split_name}_openworld_confusion_matrix.png",
    )

    roc_df = unknown_roc_curve_frame(
        pred_df["y_open_true"],
        pred_df["unknown_score"],
        unknown_id=unknown_id,
    )
    roc_df.to_csv(output_dir / f"{split_name}_unknown_roc_curve.csv", index=False)
    plot_unknown_roc_png(
        roc_df,
        output_dir / f"{split_name}_unknown_roc_curve.png",
        title=f"{split_name.upper()} unknown ROC",
    )

    save_json_clean(metrics, output_dir / f"{split_name}_openworld_metrics.json")


def evaluate_split(
    model,
    df: pd.DataFrame,
    tokenizer,
    text_col: str,
    batch_size: int,
    device,
    threshold: float | None = None,
    split_name: str = "split",
    num_workers: int = 0,
    energy_temperature: float = 1.0,
):
    if df.empty:
        raise ValueError(f"The {split_name} split is empty. Check the standardized CSV split labels.")

    unknown_id = len(KNOWN_CLASSES)
    class_names = list(KNOWN_CLASSES) + [UNKNOWN_LABEL_NAME]

    ds = OpenWorldDataset(df, tokenizer, get_eval_transform(), text_col)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logits, y_open, indices = collect_outputs(model, loader, device)
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()

    known_pred = probs.argmax(axis=1).astype(int)

    max_softmax_confidence = probs.max(axis=1)

    # Energy-based open-world thresholding from the markdown:
    # unknown if energy >= validation-selected threshold.
    unknown_score = energy_score(logits, temperature=energy_temperature)

    # OSCR expects higher confidence to mean "more likely known".
    # Since higher energy means more likely unknown, use negative energy.
    known_confidence = -unknown_score

    threshold_info = None
    if threshold is None:
        threshold_info = find_best_unknown_threshold(
            y_open_true=y_open,
            known_pred=known_pred,
            unknown_score=unknown_score,
            unknown_id=unknown_id,
        )
        threshold = threshold_info["threshold"]
    else:
        threshold_info = {
            "threshold": float(threshold),
            "energy_threshold": float(threshold),
        }

    y_open_pred = predict_open_world_from_threshold(
        known_pred=known_pred,
        unknown_score=unknown_score,
        threshold=float(threshold),
        unknown_id=unknown_id,
    )

    metrics = compute_open_world_metrics(
        y_true=y_open,
        y_pred=y_open_pred,
        unknown_score=unknown_score,
        unknown_id=unknown_id,
        known_pred=known_pred,
        known_confidence=known_confidence,
    )
    metrics.update(
        {
            "threshold_unknown_score": float(threshold),
            "threshold_energy": float(threshold),
            "energy_temperature": float(energy_temperature),
            "threshold_selection_metric": threshold_info.get("open_macro_f1"),
            "scoring_method": "energy",
            "unknown_rule": "predict UNKNOWN when energy_score >= threshold_energy",
            "text_col": text_col,
            "split": split_name,
        }
    )

    pred_df = build_prediction_dataframe(
        source_df=df,
        indices=indices,
        logits=logits,
        probs=probs,
        y_open_true=y_open,
        y_open_pred=y_open_pred,
        known_pred=known_pred,
        known_confidence=known_confidence,
        unknown_score=unknown_score,
        class_names=class_names,
        threshold=float(threshold),
        text_col=text_col,
        split_name=split_name,
    )

    return metrics, pred_df


# ============================================================
# Main
# ============================================================


def run(args):
    seed_everything(SEED)
    cfg = ExperimentConfig()
    device = get_device()

    output_dir = ensure_dir(Path(args.output_dir) / args.model_family / args.text_col)
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

    if args.text_col not in df.columns:
        raise ValueError(
            f"text_col='{args.text_col}' is missing from the standardized CSV. "
            "Expected one of: text_core, text_full, text_missing_explicit."
        )

    val_df = df[df["split"] == "target_val"].reset_index(drop=True)
    test_df = df[df["split"] == "target_test"].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    builder = build_dann_model if args.dann else build_model
    model = builder(
        args.model_family,
        image_model_name,
        cfg.text_model_name,
        len(KNOWN_CLASSES),
        cfg.fusion_dim,
        cfg.num_heads,
    ).to(device)
    load_model_state(model, args.checkpoint, strict=False)

    val_metrics, val_pred = evaluate_split(
        model=model,
        df=val_df,
        tokenizer=tokenizer,
        text_col=args.text_col,
        batch_size=args.batch_size,
        device=device,
        threshold=None,
        split_name="val",
        num_workers=getattr(cfg, "num_workers", 0),
        energy_temperature=args.energy_temperature,
    )
    test_metrics, test_pred = evaluate_split(
        model=model,
        df=test_df,
        tokenizer=tokenizer,
        text_col=args.text_col,
        batch_size=args.batch_size,
        device=device,
        threshold=val_metrics["threshold_unknown_score"],
        split_name="test",
        num_workers=getattr(cfg, "num_workers", 0),
        energy_temperature=args.energy_temperature,
    )

    class_names = list(KNOWN_CLASSES) + [UNKNOWN_LABEL_NAME]
    unknown_id = len(KNOWN_CLASSES)

    save_split_outputs("val", output_dir, val_pred, val_metrics, class_names, unknown_id)
    save_split_outputs("test", output_dir, test_pred, test_metrics, class_names, unknown_id)

    metrics_bundle = {
        "metadata": {
            "target_dataset": args.target_dataset,
            "model_family": args.model_family,
            "checkpoint": str(args.checkpoint),
            "dann": bool(args.dann),
            "text_col": args.text_col,
            "scoring_method": "energy",
            "energy_temperature": float(args.energy_temperature),
            "unknown_label_name": UNKNOWN_LABEL_NAME,
            "known_classes": list(KNOWN_CLASSES),
        },
        "val": val_metrics,
        "test": test_metrics,
    }
    save_json_clean(metrics_bundle, output_dir / "openworld_metrics.json")

    summary = pd.DataFrame([
        {"split": "val", **val_metrics},
        {"split": "test", **test_metrics},
    ])
    summary.to_csv(output_dir / "openworld_summary.csv", index=False)

    print("Saved open-world outputs to", output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardized-csv", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--target-image-roots", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-family",
        choices=[
            "mobilevit_cross_attention",
            "mobilevit_gated",
            "mobilevit_concat",
            "resnet50_cross_attention",
            "resnet50_gated",
            "resnet50_concat",
        ],
        default="mobilevit_cross_attention",
    )
    parser.add_argument(
        "--text-col",
        choices=["text_core", "text_full", "text_missing_explicit"],
        default="text_full",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--energy-temperature", type=float, default=1.0)
    parser.add_argument("--dann", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
