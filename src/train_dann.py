from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import ExperimentConfig, KNOWN_CLASSES, SEED, UNKNOWN_LABEL_NAME
from .datasets import (
    DomainAdaptationDataset,
    get_loss_weights_from_train_df,
    make_soft_weighted_random_sampler,
)
from .metrics import (
    compute_open_world_metrics,
    evaluate_closed_set,
    energy_score,
    find_best_unknown_threshold,
    predict_open_world_from_threshold,
)
from .models import build_dann_model
from .pcgrad_dann import dann_pcgrad_step  # [Shuvo Edited here]
from .preprocessing import add_known_unknown_columns, load_standardized_splits
from .transforms import get_eval_transform, get_train_transform
from .utils import ensure_dir, get_device, load_model_state, save_checkpoint, seed_everything

try:
    from .datasets import make_weighted_random_sampler
except ImportError:  # Backwards compatibility with the older datasets.py.
    make_weighted_random_sampler = None


# ============================================================
# Small IO helpers
# ============================================================

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
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    return obj


def save_json_safe(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2)


def sanitize_name(x: str) -> str:
    return str(x).replace("/", "_").replace("\\", "_").replace(" ", "_").replace("-", "_")


def model_stem(model_family: str) -> str:
    mapping = {
        "mobilevit_cross_attention": "cross_attention_mobile_vit",
        "mobilevit_gated": "gated_mobile_vit",
        "mobilevit_concat": "concat_mobile_vit",
        "resnet50_cross_attention": "cross_attention_resnet50",
        "resnet50_gated": "gated_resnet50",
        "resnet50_concat": "concat_resnet50",
    }
    return mapping.get(model_family, sanitize_name(model_family))


def metric_value(metrics: dict, *names: str, default: float = float("nan")) -> float:
    for name in names:
        if name in metrics:
            return float(metrics[name])
    return default


def label_name(label_id: int, class_names: list[str]) -> str:
    label_id = int(label_id)
    if 0 <= label_id < len(class_names):
        return class_names[label_id]
    return f"label_{label_id}"


# ============================================================
# DANN schedule and training loop
# ============================================================

def dann_lambda_schedule(step: int, total_steps: int, max_lambda: float = 1.0) -> float:
    p = step / max(total_steps, 1)
    return float(max_lambda * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))).item() - 1.0))


def train_dann_epoch(
    model,
    source_loader,
    target_loader,
    optimizer,
    class_criterion,
    domain_criterion,
    device,
    epoch: int,
    total_epochs: int,
    domain_loss_weight: float = 1.0,
    max_lambda: float = 1.0,
    use_pcgrad: bool = False,  # [Shuvo Edited here]
):
    """
    DANN training epoch aligned with the notebook behaviour.

    Important: this uses min(len(source_loader), len(target_loader)) steps.
    That matches the notebook-style loop and avoids oversampling the smaller
    loader inside a single epoch.
    """
    model.train()

    n_steps = min(len(source_loader), len(target_loader))
    if n_steps <= 0:
        raise ValueError(
            "DANN training requires non-empty source and target loaders. "
            f"Got len(source_loader)={len(source_loader)}, len(target_loader)={len(target_loader)}."
        )

    total_steps = n_steps * total_epochs
    total_loss = 0.0
    total_class_loss = 0.0
    total_domain_loss = 0.0
    total_conflicts = 0  # [Shuvo Edited here]

    for local_step, (src, tgt) in enumerate(
        tqdm(zip(source_loader, target_loader), total=n_steps, desc="dann", leave=False)
    ):
        if local_step >= n_steps:
            break

        global_step = (epoch - 1) * n_steps + local_step
        lambd = dann_lambda_schedule(global_step, total_steps, max_lambda)

        src = {k: v.to(device) if hasattr(v, "to") else v for k, v in src.items()}
        tgt = {k: v.to(device) if hasattr(v, "to") else v for k, v in tgt.items()}

        optimizer.zero_grad(set_to_none=True)

        src_class_logits, src_domain_logits, _ = model(
            src["pixel_values"],
            src.get("input_ids"),
            src.get("attention_mask"),
            dann_lambda=lambd,
        )
        _, tgt_domain_logits, _ = model(
            tgt["pixel_values"],
            tgt.get("input_ids"),
            tgt.get("attention_mask"),
            dann_lambda=lambd,
        )

        class_loss = class_criterion(src_class_logits, src["label"])

        domain_logits = torch.cat([src_domain_logits, tgt_domain_logits], dim=0)
        domain_labels = torch.cat([src["domain"], tgt["domain"]], dim=0)
        domain_loss = domain_criterion(domain_logits, domain_labels)

        # [Shuvo Edited here] gradient surgery (asymmetric PCGrad) vs. plain summed backward
        weighted_domain_loss = domain_loss_weight * domain_loss
        loss = class_loss + weighted_domain_loss
        if use_pcgrad:
            # Deconflict g_cls and g_dom BEFORE they merge; protects classification.
            stats = dann_pcgrad_step(class_loss, weighted_domain_loss, model)
            total_conflicts += stats["pcgrad_conflicts"]
        else:
            loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_class_loss += float(class_loss.item())
        total_domain_loss += float(domain_loss.item())

    return {
        "dann_loss": total_loss / n_steps,
        "class_loss": total_class_loss / n_steps,
        "domain_loss": total_domain_loss / n_steps,
        "n_steps": n_steps,
        "pcgrad_conflicts_per_step": total_conflicts / n_steps,  # [Shuvo Edited here]
    }


# ============================================================
# Loaders and samplers
# ============================================================

def build_sampler(df: pd.DataFrame, sampler_name: str, cfg: ExperimentConfig, label_col: str = "label_id"):
    if sampler_name == "none":
        return None

    if sampler_name == "notebook":
        if make_weighted_random_sampler is not None:
            return make_weighted_random_sampler(df, label_col=label_col, seed=SEED)
        print("make_weighted_random_sampler() not found; falling back to soft weighted sampler.")

    return make_soft_weighted_random_sampler(df, beta=cfg.balance_beta, label_col=label_col)


def make_loader(dataset, batch_size: int, num_workers: int, sampler=None, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else shuffle,
        num_workers=num_workers,
    )


# ============================================================
# Closed-set known-only evaluation artifacts
# ============================================================

def save_closed_set_artifacts(
    *,
    output_dir: Path,
    split_name: str,
    metrics: dict,
    y_true,
    y_pred,
    y_prob,
    class_names: list[str],
):
    split_dir = ensure_dir(output_dir / split_name)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    save_json_safe(metrics, split_dir / "metrics.json")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(split_dir / "classification_report.csv")
    save_json_safe(report, split_dir / "classification_report.json")

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_true_label": [label_name(x, class_names) for x in y_true],
        "y_pred": y_pred,
        "y_pred_label": [label_name(x, class_names) for x in y_pred],
    })
    for i, name in enumerate(class_names):
        pred_df[f"prob_{name}"] = y_prob[:, i]
    pred_df.to_csv(split_dir / "predictions.csv", index=False)

    cm_count = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize="true")

    pd.DataFrame(cm_count, index=class_names, columns=class_names).to_csv(split_dir / "confusion_matrix_counts.csv")
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(split_dir / "confusion_matrix_normalized.csv")

    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    disp.plot(ax=ax, values_format=".4f", xticks_rotation=45, colorbar=True)
    ax.set_title(split_name.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(split_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 7))
    roc_rows = []

    for i, name in enumerate(class_names):
        if y_bin.shape[1] <= i or len(np.unique(y_bin[:, i])) < 2:
            continue

        fpr, tpr, thresholds = roc_curve(y_bin[:, i], y_prob[:, i])
        class_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} AUC={class_auc:.4f}")

        roc_rows.extend(
            {
                "class": name,
                "fpr": float(fp),
                "tpr": float(tp),
                "threshold": float(th),
                "auc": float(class_auc),
            }
            for fp, tp, th in zip(fpr, tpr, thresholds)
        )

    if roc_rows:
        pd.DataFrame(roc_rows).to_csv(split_dir / "roc_curve_data.csv", index=False)
        ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(f"{split_name.replace('_', ' ').title()} ROC")
        fig.tight_layout()
        fig.savefig(split_dir / "roc_curves.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


def evaluate_known_split(
    *,
    model,
    df: pd.DataFrame,
    tokenizer,
    text_col: str,
    batch_size: int,
    device,
    criterion,
    cfg: ExperimentConfig,
    output_dir: Path,
    split_name: str,
):
    if df.empty:
        print(f"Skipping {split_name}: no known-class rows.")
        return None

    dataset = DomainAdaptationDataset(
        df,
        tokenizer,
        get_eval_transform(),
        text_col,
        domain_label=1,
        max_text_len=cfg.max_text_len,
    )
    loader = make_loader(dataset, batch_size, cfg.num_workers, sampler=None, shuffle=False)
    metrics, y_true, y_pred, y_prob = evaluate_closed_set(
        model,
        loader,
        criterion,
        device,
        len(KNOWN_CLASSES),
    )
    save_closed_set_artifacts(
        output_dir=output_dir,
        split_name=split_name,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=list(KNOWN_CLASSES),
    )
    return metrics


# ============================================================
# Open-world evaluation artifacts
# ============================================================

@torch.no_grad()
def collect_logits(model, loader, device):
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
        raise ValueError("Cannot evaluate open-world split because the loader is empty.")

    return np.concatenate(logits_all, axis=0), np.asarray(labels_all), np.asarray(indices_all)


def evaluate_open_world_split(
    *,
    model,
    df: pd.DataFrame,
    tokenizer,
    text_col: str,
    batch_size: int,
    device,
    cfg: ExperimentConfig,
    threshold: float | None = None,
    energy_temperature: float = 1.0,
):
    dataset = DomainAdaptationDataset(
        df,
        tokenizer,
        get_eval_transform(),
        text_col,
        label_col="label_open_id",
        domain_label=None,
        max_text_len=cfg.max_text_len,
    )
    loader = make_loader(dataset, batch_size, cfg.num_workers, sampler=None, shuffle=False)

    logits, y_open, indices = collect_logits(model, loader, device)

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    known_pred = probs.argmax(axis=1)
    max_softmax_confidence = probs.max(axis=1)

    # Energy-based open-world thresholding from the markdown:
    # unknown if energy >= validation-selected threshold.
    unknown_score = energy_score(logits, temperature=energy_temperature)

    # OSCR expects higher confidence to mean "more likely known".
    # Since higher energy means more likely unknown, use negative energy.
    known_confidence = -unknown_score

    unknown_id = len(KNOWN_CLASSES)
    if threshold is None:
        best = find_best_unknown_threshold(y_open, known_pred, unknown_score, unknown_id)
        threshold = float(best["threshold"])

    y_open_pred = predict_open_world_from_threshold(
        known_pred=known_pred,
        unknown_score=unknown_score,
        threshold=threshold,
        unknown_id=unknown_id,
    )

    try:
        metrics = compute_open_world_metrics(
            y_open,
            y_open_pred,
            unknown_score,
            unknown_id,
            known_pred=known_pred,
            known_confidence=known_confidence,
        )
    except TypeError:
        metrics = compute_open_world_metrics(y_open, y_open_pred, unknown_score, unknown_id)

    metrics["threshold_unknown_score"] = float(threshold)
    metrics["threshold_energy"] = float(threshold)
    metrics["energy_temperature"] = float(energy_temperature)
    metrics["scoring_method"] = "energy"
    metrics["threshold_rule"] = "predict UNKNOWN when energy_score >= threshold_energy"

    pred_df = df.iloc[indices].copy().reset_index(drop=True)
    class_names = list(KNOWN_CLASSES) + [UNKNOWN_LABEL_NAME]

    pred_df["y_open_true"] = y_open
    pred_df["y_open_true_label"] = [label_name(x, class_names) for x in y_open]
    pred_df["known_pred"] = known_pred
    pred_df["known_pred_label"] = [label_name(x, list(KNOWN_CLASSES)) for x in known_pred]
    pred_df["known_confidence"] = known_confidence
    pred_df["max_softmax_confidence"] = max_softmax_confidence
    pred_df["energy_score"] = unknown_score
    pred_df["unknown_score"] = unknown_score
    pred_df["y_open_pred"] = y_open_pred
    pred_df["y_open_pred_label"] = [label_name(x, class_names) for x in y_open_pred]

    for i, name in enumerate(KNOWN_CLASSES):
        pred_df[f"logit_{name}"] = logits[:, i]
        pred_df[f"prob_{name}"] = probs[:, i]

    return metrics, pred_df


def save_open_world_artifacts(
    *,
    output_dir: Path,
    split_name: str,
    metrics: dict,
    pred_df: pd.DataFrame,
):
    split_dir = ensure_dir(output_dir / split_name)
    class_names = list(KNOWN_CLASSES) + [UNKNOWN_LABEL_NAME]
    labels = list(range(len(class_names)))
    unknown_id = len(KNOWN_CLASSES)

    save_json_safe(metrics, split_dir / "metrics.json")
    pred_df.to_csv(split_dir / "predictions.csv", index=False)

    y_true = pred_df["y_open_true"].to_numpy()
    y_pred = pred_df["y_open_pred"].to_numpy()
    unknown_score = pred_df["unknown_score"].to_numpy()

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
    save_json_safe(report, split_dir / "classification_report.json")

    cm_count = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    pd.DataFrame(cm_count, index=class_names, columns=class_names).to_csv(split_dir / "confusion_matrix_counts.csv")
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(split_dir / "confusion_matrix_normalized.csv")

    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    disp.plot(ax=ax, values_format=".4f", xticks_rotation=45, colorbar=True)
    ax.set_title(split_name.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(split_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    is_unknown = (y_true == unknown_id).astype(int)
    if len(np.unique(is_unknown)) == 2:
        fpr, tpr, thresholds = roc_curve(is_unknown, unknown_score)
        roc_auc = auc(fpr, tpr)

        pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
            "auc": roc_auc,
        }).to_csv(split_dir / "unknown_roc_curve.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f"Unknown AUROC={roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{split_name.replace('_', ' ').title()} Unknown ROC")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(split_dir / "unknown_roc_curve.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# Main run
# ============================================================

def run(args):
    seed_everything(SEED)

    cfg = ExperimentConfig()
    device = get_device()

    target_name = sanitize_name(args.target_dataset)
    stem = model_stem(args.model_family)

    output_dir = ensure_dir(
        Path(args.output_dir)
        / args.model_family
        / args.text_col
        / target_name
    )

    image_model_name = (
        cfg.resnet_model_name
        if args.model_family.startswith("resnet50")
        else cfg.image_model_name
    )

    source_df = load_standardized_splits(
        args.standardized_csv,
        args.source_image_roots,
        args.source_dataset,
    )
    target_df = load_standardized_splits(
        args.standardized_csv,
        args.target_image_roots,
        args.target_dataset,
    )

    source_df = add_known_unknown_columns(source_df)
    target_df = add_known_unknown_columns(target_df)

    source_train_df = source_df[
        (source_df["split"].isin(["source_train", "train"]))
        & (~source_df["is_unknown"])
    ].copy()

    target_adapt_df = target_df[
        (target_df["split"] == "target_adapt")
        & (~target_df["is_unknown"])
    ].copy()

    target_val_df = target_df[target_df["split"] == "target_val"].copy()
    target_test_df = target_df[target_df["split"] == "target_test"].copy()

    target_val_known_df = target_val_df[~target_val_df["is_unknown"]].copy()
    target_test_known_df = target_test_df[~target_test_df["is_unknown"]].copy()

    if source_train_df.empty:
        raise ValueError("No known-class source training rows found.")
    if target_adapt_df.empty:
        raise ValueError("No known-class target adaptation rows found.")
    if target_val_known_df.empty:
        raise ValueError("No known-class target validation rows found.")

    split_sizes = {
        "source_train_known": int(len(source_train_df)),
        "target_adapt_known": int(len(target_adapt_df)),
        "target_val_all": int(len(target_val_df)),
        "target_val_known": int(len(target_val_known_df)),
        "target_test_all": int(len(target_test_df)),
        "target_test_known": int(len(target_test_known_df)),
    }
    save_json_safe(split_sizes, output_dir / "split_sizes.json")

    source_train_df.to_csv(output_dir / "source_train_known_split.csv", index=False)
    target_adapt_df.to_csv(output_dir / "target_adapt_known_split.csv", index=False)
    target_val_df.to_csv(output_dir / "target_val_all_split.csv", index=False)
    target_test_df.to_csv(output_dir / "target_test_all_split.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    source_ds = DomainAdaptationDataset(
        source_train_df,
        tokenizer,
        get_train_transform(),
        args.text_col,
        domain_label=0,
        max_text_len=cfg.max_text_len,
    )
    target_ds = DomainAdaptationDataset(
        target_adapt_df,
        tokenizer,
        get_train_transform(),
        args.text_col,
        domain_label=1,
        max_text_len=cfg.max_text_len,
    )
    val_known_ds = DomainAdaptationDataset(
        target_val_known_df,
        tokenizer,
        get_eval_transform(),
        args.text_col,
        domain_label=1,
        max_text_len=cfg.max_text_len,
    )

    source_sampler = build_sampler(source_train_df, args.sampler, cfg)
    target_sampler = build_sampler(target_adapt_df, args.sampler, cfg)

    source_loader = make_loader(
        source_ds,
        args.batch_size,
        cfg.num_workers,
        sampler=source_sampler,
        shuffle=source_sampler is None,
    )
    target_loader = make_loader(
        target_ds,
        args.batch_size,
        cfg.num_workers,
        sampler=target_sampler,
        shuffle=target_sampler is None,
    )
    val_known_loader = make_loader(
        val_known_ds,
        args.batch_size,
        cfg.num_workers,
        sampler=None,
        shuffle=False,
    )

    model = build_dann_model(
        args.model_family,
        image_model_name,
        cfg.text_model_name,
        len(KNOWN_CLASSES),
        cfg.fusion_dim,
        cfg.num_heads,
    ).to(device)

    loaded_checkpoint_info = {}
    if args.checkpoint:
        missing, unexpected = load_model_state(model, args.checkpoint, strict=False)
        loaded_checkpoint_info = {
            "checkpoint": str(args.checkpoint),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
        save_json_safe(loaded_checkpoint_info, output_dir / "loaded_checkpoint_info.json")

    class_weights = (
        get_loss_weights_from_train_df(source_train_df, len(KNOWN_CLASSES), cfg.balance_beta).to(device)
        if getattr(cfg, "use_weighted_loss", True)
        else None
    )
    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    domain_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    # --------------------------------------------------------
    # Before-DANN known-only target evaluation
    # --------------------------------------------------------
    before_metrics = {}
    before_val = evaluate_known_split(
        model=model,
        df=target_val_known_df,
        tokenizer=tokenizer,
        text_col=args.text_col,
        batch_size=args.batch_size,
        device=device,
        criterion=class_criterion,
        cfg=cfg,
        output_dir=output_dir,
        split_name="before_dann_target_val_known",
    )
    if before_val is not None:
        before_metrics["before_dann_target_val_known"] = before_val

    before_test = evaluate_known_split(
        model=model,
        df=target_test_known_df,
        tokenizer=tokenizer,
        text_col=args.text_col,
        batch_size=args.batch_size,
        device=device,
        criterion=class_criterion,
        cfg=cfg,
        output_dir=output_dir,
        split_name="before_dann_target_test_known",
    )
    if before_test is not None:
        before_metrics["before_dann_target_test_known"] = before_test

    save_json_safe(before_metrics, output_dir / "before_dann_metrics.json")

    # --------------------------------------------------------
    # DANN training with early stopping
    # --------------------------------------------------------
    best_val_f1 = -1.0
    best_epoch = 0
    patience_left = args.patience if args.patience is not None else cfg.patience
    history = []

    best_ckpt = output_dir / "best_dann.pt"
    best_named_ckpt = output_dir / f"{stem}_{target_name}_{args.text_col}_best_dann.pt"
    final_ckpt = output_dir / "final_dann.pt"
    final_named_ckpt = output_dir / f"{stem}_{target_name}_{args.text_col}_final_dann.pt"

    for epoch in range(1, args.epochs + 1):
        train_stats = train_dann_epoch(
            model=model,
            source_loader=source_loader,
            target_loader=target_loader,
            optimizer=optimizer,
            class_criterion=class_criterion,
            domain_criterion=domain_criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            domain_loss_weight=cfg.domain_loss_weight,
            max_lambda=cfg.dann_lambda_max,
            use_pcgrad=args.pcgrad,  # [Shuvo Edited here]
        )

        val_metrics, _, _, _ = evaluate_closed_set(
            model,
            val_known_loader,
            class_criterion,
            device,
            len(KNOWN_CLASSES),
        )

        val_f1 = metric_value(val_metrics, "macro_f1", "f1_macro")

        row = {
            "epoch": epoch,
            **train_stats,
            **{f"target_val_known_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(row)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_left = args.patience if args.patience is not None else cfg.patience

            extra = {
                "epoch": epoch,
                "model_family": args.model_family,
                "text_col": args.text_col,
                "source_dataset": args.source_dataset,
                "target_dataset": args.target_dataset,
                "known_classes": list(KNOWN_CLASSES),
                "best_target_val_macro_f1": float(best_val_f1),
                "checkpoint_role": "best_dann",
            }
            save_checkpoint(model, best_ckpt, extra)
            save_checkpoint(model, best_named_ckpt, extra)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
                break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    save_checkpoint(
        model,
        final_ckpt,
        {
            "epoch": history[-1]["epoch"] if history else 0,
            "model_family": args.model_family,
            "text_col": args.text_col,
            "source_dataset": args.source_dataset,
            "target_dataset": args.target_dataset,
            "known_classes": list(KNOWN_CLASSES),
            "checkpoint_role": "final_dann",
        },
    )
    save_checkpoint(
        model,
        final_named_ckpt,
        {
            "epoch": history[-1]["epoch"] if history else 0,
            "model_family": args.model_family,
            "text_col": args.text_col,
            "source_dataset": args.source_dataset,
            "target_dataset": args.target_dataset,
            "known_classes": list(KNOWN_CLASSES),
            "checkpoint_role": "final_dann",
        },
    )

    # --------------------------------------------------------
    # Load best DANN and evaluate after-DANN known-only splits
    # --------------------------------------------------------
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    after_metrics = {}
    after_val = evaluate_known_split(
        model=model,
        df=target_val_known_df,
        tokenizer=tokenizer,
        text_col=args.text_col,
        batch_size=args.batch_size,
        device=device,
        criterion=class_criterion,
        cfg=cfg,
        output_dir=output_dir,
        split_name="after_dann_target_val_known",
    )
    if after_val is not None:
        after_metrics["after_dann_target_val_known"] = after_val

    after_test = evaluate_known_split(
        model=model,
        df=target_test_known_df,
        tokenizer=tokenizer,
        text_col=args.text_col,
        batch_size=args.batch_size,
        device=device,
        criterion=class_criterion,
        cfg=cfg,
        output_dir=output_dir,
        split_name="after_dann_target_test_known",
    )
    if after_test is not None:
        after_metrics["after_dann_target_test_known"] = after_test

    save_json_safe(after_metrics, output_dir / "after_dann_known_metrics.json")

    # --------------------------------------------------------
    # Open-world evaluation loop: val threshold -> test result
    # --------------------------------------------------------
    openworld_metrics = {}
    if not args.skip_openworld:
        if target_val_df.empty:
            print("Skipping open-world evaluation: target_val split is empty.")
        else:
            val_open_metrics, val_open_pred = evaluate_open_world_split(
                model=model,
                df=target_val_df.reset_index(drop=True),
                tokenizer=tokenizer,
                text_col=args.text_col,
                batch_size=args.batch_size,
                device=device,
                cfg=cfg,
                threshold=None,
                energy_temperature=args.energy_temperature,
            )
            save_open_world_artifacts(
                output_dir=output_dir,
                split_name="openworld_target_val",
                metrics=val_open_metrics,
                pred_df=val_open_pred,
            )
            openworld_metrics["openworld_target_val"] = val_open_metrics

            if target_test_df.empty:
                print("Skipping target-test open-world evaluation: target_test split is empty.")
            else:
                test_open_metrics, test_open_pred = evaluate_open_world_split(
                    model=model,
                    df=target_test_df.reset_index(drop=True),
                    tokenizer=tokenizer,
                    text_col=args.text_col,
                    batch_size=args.batch_size,
                    device=device,
                    cfg=cfg,
                    threshold=val_open_metrics["threshold_unknown_score"],
                    energy_temperature=args.energy_temperature,
                )
                save_open_world_artifacts(
                    output_dir=output_dir,
                    split_name="openworld_target_test",
                    metrics=test_open_metrics,
                    pred_df=test_open_pred,
                )
                openworld_metrics["openworld_target_test"] = test_open_metrics

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    summary = {
        "model_family": args.model_family,
        "text_col": args.text_col,
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "output_dir": str(output_dir),
        "loaded_checkpoint": loaded_checkpoint_info,
        "split_sizes": split_sizes,
        "best_epoch": best_epoch,
        "best_target_val_macro_f1": best_val_f1,
        "best_checkpoint": str(best_ckpt),
        "best_named_checkpoint": str(best_named_ckpt),
        "final_checkpoint": str(final_ckpt),
        "final_named_checkpoint": str(final_named_ckpt),
        "before_dann": before_metrics,
        "after_dann": after_metrics,
        "openworld": openworld_metrics,
    }

    save_json_safe(summary, output_dir / "metrics.json")
    save_json_safe(summary, output_dir / "dann_summary.json")

    flat_rows = []
    for group_name, group in [
        ("before_dann", before_metrics),
        ("after_dann", after_metrics),
        ("openworld", openworld_metrics),
    ]:
        for split_name, metrics in group.items():
            flat_rows.append({
                "group": group_name,
                "split": split_name,
                "model_family": args.model_family,
                "text_col": args.text_col,
                "source_dataset": args.source_dataset,
                "target_dataset": args.target_dataset,
                **metrics,
            })

    if flat_rows:
        pd.DataFrame(flat_rows).to_csv(output_dir / "dann_metrics_summary.csv", index=False)

    print("Saved DANN outputs to", output_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--standardized-csv", required=True)
    parser.add_argument("--source-dataset", default="PAD-UFES")
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--source-image-roots", nargs="+", required=True)
    parser.add_argument("--target-image-roots", nargs="+", required=True)
    parser.add_argument("--checkpoint")
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=None)

    parser.add_argument(
        "--sampler",
        choices=["notebook", "soft", "none"],
        default="notebook",
        help=(
            "'notebook' uses inverse class-frequency WeightedRandomSampler; "
            "'soft' uses effective-number weighting; 'none' disables sampling."
        ),
    )

    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--energy-temperature", type=float, default=1.0)
    parser.add_argument("--skip-openworld", action="store_true")

    # [Shuvo Edited here] enable asymmetric PCGrad gradient surgery on the DANN step
    parser.add_argument(
        "--pcgrad",
        action="store_true",
        help=(
            "Deconflict the classification and domain gradients (asymmetric "
            "PCGrad) before each optimizer step, protecting classification from "
            "DANN negative transfer. Off by default = original summed-loss DANN."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
