from __future__ import annotations

from typing import Iterable, Sequence

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
from sklearn.preprocessing import label_binarize


# ============================================================
# Helpers
# ============================================================


def _as_numpy(x):
    return np.asarray(x)


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _metric_aliases(metrics: dict) -> dict:
    """Add both notebook-style and package-style metric names.

    Original notebook style uses names such as macro_f1 and macro_auc_ovr.
    Earlier modular code used names such as f1_macro and macro_auc.
    Keeping both prevents downstream scripts from breaking.
    """

    alias_pairs = {
        "macro_precision": "precision_macro",
        "macro_recall": "recall_macro",
        "macro_f1": "f1_macro",
        "weighted_precision": "precision_weighted",
        "weighted_recall": "recall_weighted",
        "weighted_f1": "f1_weighted",
        "macro_auc_ovr": "macro_auc",
        "weighted_auc_ovr": "weighted_auc",
    }

    for notebook_name, package_name in alias_pairs.items():
        if notebook_name in metrics and package_name not in metrics:
            metrics[package_name] = metrics[notebook_name]
        if package_name in metrics and notebook_name not in metrics:
            metrics[notebook_name] = metrics[package_name]

    if "macro_auc_ovr" in metrics and "auc" not in metrics:
        metrics["auc"] = metrics["macro_auc_ovr"]

    return metrics


# ============================================================
# Closed-set metrics
# ============================================================


def compute_multiclass_auc(
    y_true,
    y_prob,
    labels: Sequence[int],
    class_names: Sequence[str] | None = None,
) -> dict:
    """Compute multiclass one-vs-rest AUC metrics.

    Returns notebook-compatible keys:
    - macro_auc_ovr
    - weighted_auc_ovr
    - auc, as an alias of macro_auc_ovr

    Also returns backwards-compatible package aliases:
    - macro_auc
    - weighted_auc

    Per-class AUC keys are added as auc_<class_name> when class_names is given.
    """

    y_true = _as_numpy(y_true)
    y_prob = _as_numpy(y_prob)
    labels = list(labels)

    metrics = {
        "macro_auc_ovr": np.nan,
        "weighted_auc_ovr": np.nan,
    }

    if y_prob.ndim != 2 or y_prob.shape[0] != len(y_true) or len(np.unique(y_true)) < 2:
        return _metric_aliases(metrics)

    try:
        metrics["macro_auc_ovr"] = float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        metrics["macro_auc_ovr"] = np.nan

    try:
        metrics["weighted_auc_ovr"] = float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=labels,
                multi_class="ovr",
                average="weighted",
            )
        )
    except Exception:
        metrics["weighted_auc_ovr"] = np.nan

    if class_names is not None:
        y_bin = label_binarize(y_true, classes=labels)
        class_names = list(class_names)

        for class_pos, class_name in enumerate(class_names):
            key = f"auc_{class_name}"
            try:
                if class_pos >= y_bin.shape[1] or class_pos >= y_prob.shape[1]:
                    metrics[key] = np.nan
                elif len(np.unique(y_bin[:, class_pos])) < 2:
                    metrics[key] = np.nan
                else:
                    metrics[key] = float(
                        roc_auc_score(y_bin[:, class_pos], y_prob[:, class_pos])
                    )
            except Exception:
                metrics[key] = np.nan

    return _metric_aliases(metrics)


def compute_closed_set_metrics(
    y_true,
    y_pred,
    y_prob=None,
    labels: Sequence[int] | None = None,
    class_names: Sequence[str] | None = None,
    avg_loss: float | None = None,
) -> dict:
    """Compute closed-set classification metrics.

    The returned dictionary intentionally includes both naming conventions:
    - notebook style: macro_precision, macro_recall, macro_f1, weighted_f1
    - earlier module style: precision_macro, recall_macro, f1_macro, f1_weighted
    """

    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    if avg_loss is not None:
        metrics["loss"] = _safe_float(avg_loss)

    if y_prob is not None and labels is not None:
        metrics.update(
            compute_multiclass_auc(
                y_true=y_true,
                y_prob=y_prob,
                labels=labels,
                class_names=class_names,
            )
        )

    return _metric_aliases(metrics)


@torch.no_grad()
def evaluate_closed_set(model, loader, criterion, device, num_classes: int, class_names=None):
    model.eval()
    losses, y_true, y_pred, y_prob = [], [], [], []

    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        out = model(batch["pixel_values"], batch.get("input_ids"), batch.get("attention_mask"))
        logits = out[0] if isinstance(out, tuple) else out
        loss = criterion(logits, batch["label"])
        probs = F.softmax(logits, dim=1)

        losses.append(float(loss.item()))
        y_true.extend(batch["label"].detach().cpu().numpy().tolist())
        y_pred.extend(torch.argmax(probs, dim=1).detach().cpu().numpy().tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())

    labels = list(range(num_classes))
    metrics = compute_closed_set_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=np.asarray(y_prob),
        labels=labels,
        class_names=class_names,
        avg_loss=float(np.mean(losses)) if losses else np.nan,
    )

    return metrics, np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)


# ============================================================
# Open-world scores and thresholding
# ============================================================


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Energy score where lower values often indicate confident known predictions."""

    logits = np.asarray(logits, dtype=float)
    scaled = logits / temperature
    max_logits = np.max(scaled, axis=1, keepdims=True)
    logsumexp = max_logits.squeeze(1) + np.log(
        np.exp(scaled - max_logits).sum(axis=1)
    )
    return -temperature * logsumexp


def max_softmax_unknown_score(probs: np.ndarray) -> np.ndarray:
    """Notebook-style unknownness score: 1 - max softmax confidence."""

    probs = np.asarray(probs)
    return 1.0 - np.max(probs, axis=1)


def predict_open_world_from_threshold(
    known_pred: np.ndarray,
    unknown_score: np.ndarray,
    threshold: float,
    unknown_id: int,
):
    """Assign unknown_id when unknown_score is greater than or equal to threshold."""

    pred = np.asarray(known_pred).copy()
    pred[np.asarray(unknown_score) >= threshold] = unknown_id
    return pred


def find_best_unknown_threshold(
    y_open_true,
    known_pred,
    unknown_score,
    unknown_id: int,
    grid_size: int = 400,
    objective: str = "macro_f1",
):
    """Find the threshold that maximizes open-world macro F1 by default."""

    scores = np.asarray(unknown_score, dtype=float)
    if scores.size == 0:
        return {"threshold": np.nan, objective: np.nan, "macro_f1": np.nan}

    if np.allclose(scores.min(), scores.max()):
        thresholds = np.asarray([scores.min()])
    else:
        thresholds = np.linspace(scores.min(), scores.max(), grid_size)

    best = {"threshold": float(thresholds[0]), objective: -1.0, "macro_f1": -1.0}

    for threshold in thresholds:
        pred = predict_open_world_from_threshold(
            known_pred=known_pred,
            unknown_score=scores,
            threshold=float(threshold),
            unknown_id=unknown_id,
        )
        metrics = compute_open_world_metrics(
            y_open_true=y_open_true,
            y_open_pred=pred,
            unknown_score=scores,
            unknown_id=unknown_id,
            known_pred=known_pred,
            known_confidence=-scores,
        )
        score = metrics.get(objective, metrics.get("open_macro_f1", np.nan))
        if np.isnan(score):
            continue
        if score > best[objective]:
            best = {
                "threshold": float(threshold),
                objective: float(score),
                "macro_f1": float(metrics.get("open_macro_f1", score)),
            }

    return best


# ============================================================
# Open-world metrics
# ============================================================


def compute_oscr(
    y_true_known_unknown,
    y_true_known_labels,
    known_pred,
    known_confidence,
) -> float:
    """Compute OSCR: area under CCR-vs-FPR curve.

    Parameters
    ----------
    y_true_known_unknown:
        Boolean array. True for known samples, False for unknown samples.
    y_true_known_labels:
        True known class IDs. Values for unknown rows are ignored.
    known_pred:
        Closed-set predicted known class IDs before applying the unknown threshold.
    known_confidence:
        Higher values must mean more likely known / more confidently accepted.
    """

    y_true_known_unknown = np.asarray(y_true_known_unknown).astype(bool)
    y_true_known_labels = np.asarray(y_true_known_labels)
    known_pred = np.asarray(known_pred)
    known_confidence = np.asarray(known_confidence, dtype=float)

    if len(known_confidence) == 0:
        return np.nan

    thresholds = np.sort(np.unique(known_confidence))[::-1]
    ccr, fpr = [], []
    n_known = max(int(y_true_known_unknown.sum()), 1)
    n_unknown = max(int((~y_true_known_unknown).sum()), 1)

    for threshold in thresholds:
        accept = known_confidence >= threshold
        correct_known = accept & y_true_known_unknown & (known_pred == y_true_known_labels)
        false_known = accept & (~y_true_known_unknown)
        ccr.append(correct_known.sum() / n_known)
        fpr.append(false_known.sum() / n_unknown)

    if len(fpr) < 2:
        return np.nan

    order = np.argsort(fpr)
    return float(np.trapz(np.asarray(ccr)[order], np.asarray(fpr)[order]))


def compute_open_world_metrics(
    y_open_true,
    y_open_pred,
    unknown_score,
    unknown_id: int,
    known_pred=None,
    known_confidence=None,
) -> dict:
    """Compute notebook-style open-world metrics.

    Includes:
    - open accuracy
    - open macro precision/recall/F1
    - open weighted precision/recall/F1
    - unknown precision/recall/F1
    - unknown AUROC
    - optional OSCR when known_pred / known_confidence are available
    """

    y_open_true = np.asarray(y_open_true)
    y_open_pred = np.asarray(y_open_pred)
    unknown_score = np.asarray(unknown_score, dtype=float)

    true_unknown = y_open_true == unknown_id
    pred_unknown = y_open_pred == unknown_id

    metrics = {
        "open_accuracy": float(accuracy_score(y_open_true, y_open_pred)),
        "open_macro_precision": float(
            precision_score(y_open_true, y_open_pred, average="macro", zero_division=0)
        ),
        "open_macro_recall": float(
            recall_score(y_open_true, y_open_pred, average="macro", zero_division=0)
        ),
        "open_macro_f1": float(
            f1_score(y_open_true, y_open_pred, average="macro", zero_division=0)
        ),
        "open_weighted_precision": float(
            precision_score(y_open_true, y_open_pred, average="weighted", zero_division=0)
        ),
        "open_weighted_recall": float(
            recall_score(y_open_true, y_open_pred, average="weighted", zero_division=0)
        ),
        "open_weighted_f1": float(
            f1_score(y_open_true, y_open_pred, average="weighted", zero_division=0)
        ),
        "unknown_precision": float(
            precision_score(true_unknown.astype(int), pred_unknown.astype(int), zero_division=0)
        ),
        "unknown_recall": float(
            recall_score(true_unknown.astype(int), pred_unknown.astype(int), zero_division=0)
        ),
        "unknown_f1": float(
            f1_score(true_unknown.astype(int), pred_unknown.astype(int), zero_division=0)
        ),
    }

    if len(np.unique(true_unknown.astype(int))) == 2:
        try:
            metrics["unknown_auroc"] = float(
                roc_auc_score(true_unknown.astype(int), unknown_score)
            )
        except Exception:
            metrics["unknown_auroc"] = np.nan
    else:
        metrics["unknown_auroc"] = np.nan

    # Backwards-compatible shorthand aliases.
    metrics["macro_f1"] = metrics["open_macro_f1"]
    metrics["weighted_f1"] = metrics["open_weighted_f1"]

    if known_pred is None:
        # Fallback only. For a cleaner OSCR, pass the raw closed-set known_pred
        # before thresholding unknown samples.
        known_pred = np.where(pred_unknown, -1, y_open_pred)

    if known_confidence is None and unknown_score is not None:
        # unknown_score is higher for unknown, so negative score is higher for known.
        known_confidence = -unknown_score

    if known_pred is not None and known_confidence is not None:
        try:
            metrics["oscr"] = compute_oscr(
                y_true_known_unknown=~true_unknown,
                y_true_known_labels=y_open_true,
                known_pred=known_pred,
                known_confidence=known_confidence,
            )
        except Exception:
            metrics["oscr"] = np.nan
    else:
        metrics["oscr"] = np.nan

    return metrics


# ============================================================
# Report / output helpers used by training and evaluation scripts
# ============================================================


def classification_report_dict(
    y_true,
    y_pred,
    labels: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    return classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        target_names=list(class_names),
        zero_division=0,
        digits=4,
        output_dict=True,
    )


def classification_report_frame(
    y_true,
    y_pred,
    labels: Sequence[int],
    class_names: Sequence[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        classification_report_dict(y_true, y_pred, labels, class_names)
    ).transpose()


def prediction_dataframe(
    y_true,
    y_pred,
    y_prob=None,
    class_names: Sequence[str] | None = None,
    unknown_score=None,
    known_pred=None,
    extra_columns: dict | None = None,
) -> pd.DataFrame:
    data = {
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
    }

    if class_names is not None:
        names = list(class_names)
        data["y_true_label"] = [names[int(x)] if 0 <= int(x) < len(names) else "UNKNOWN" for x in y_true]
        data["y_pred_label"] = [names[int(x)] if 0 <= int(x) < len(names) else "UNKNOWN" for x in y_pred]

    if known_pred is not None:
        data["known_pred"] = np.asarray(known_pred)
        if class_names is not None:
            names = list(class_names)
            data["known_pred_label"] = [
                names[int(x)] if 0 <= int(x) < len(names) else "UNKNOWN"
                for x in known_pred
            ]

    if unknown_score is not None:
        data["unknown_score"] = np.asarray(unknown_score)

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if class_names is None:
            prob_names = [f"class_{i}" for i in range(y_prob.shape[1])]
        else:
            prob_names = list(class_names)[: y_prob.shape[1]]
        for i, name in enumerate(prob_names):
            data[f"prob_{name}"] = y_prob[:, i]

    if extra_columns:
        for key, value in extra_columns.items():
            data[key] = value

    return pd.DataFrame(data)


def confusion_matrix_frame(
    y_true,
    y_pred,
    labels: Sequence[int],
    class_names: Sequence[str],
    normalize: str | None = None,
) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels), normalize=normalize)
    return pd.DataFrame(cm, index=list(class_names), columns=list(class_names))


def unknown_roc_curve_data(y_open_true, unknown_score, unknown_id: int) -> dict:
    y_open_true = np.asarray(y_open_true)
    unknown_score = np.asarray(unknown_score, dtype=float)
    is_unknown = (y_open_true == unknown_id).astype(int)

    if len(np.unique(is_unknown)) < 2:
        return {
            "fpr": np.asarray([]),
            "tpr": np.asarray([]),
            "thresholds": np.asarray([]),
            "unknown_auroc": np.nan,
        }

    fpr, tpr, thresholds = roc_curve(is_unknown, unknown_score)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "unknown_auroc": float(auc(fpr, tpr)),
    }
