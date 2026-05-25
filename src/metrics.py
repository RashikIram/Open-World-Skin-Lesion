from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def compute_multiclass_auc(y_true, y_prob, labels):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    present = np.unique(y_true)
    if len(present) < 2:
        return {"macro_auc": np.nan, "weighted_auc": np.nan}
    try:
        y_bin = label_binarize(y_true, classes=labels)
        return {
            "macro_auc": float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")),
            "weighted_auc": float(roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")),
        }
    except Exception:
        return {"macro_auc": np.nan, "weighted_auc": np.nan}


def compute_closed_set_metrics(y_true, y_pred, y_prob=None, labels=None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_prob is not None and labels is not None:
        metrics.update(compute_multiclass_auc(y_true, y_prob, labels))
    return metrics


@torch.no_grad()
def evaluate_closed_set(model, loader, criterion, device, num_classes: int):
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
    metrics = compute_closed_set_metrics(y_true, y_pred, np.asarray(y_prob), list(range(num_classes)))
    metrics["loss"] = float(np.mean(losses)) if losses else np.nan
    return metrics, np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits)
    return -temperature * np.log(np.exp(logits / temperature).sum(axis=1))


def max_softmax_unknown_score(probs: np.ndarray) -> np.ndarray:
    return 1.0 - np.max(probs, axis=1)


def predict_open_world_from_threshold(known_pred: np.ndarray, unknown_score: np.ndarray, threshold: float, unknown_id: int):
    pred = np.asarray(known_pred).copy()
    pred[np.asarray(unknown_score) >= threshold] = unknown_id
    return pred


def find_best_unknown_threshold(y_open_true, known_pred, unknown_score, unknown_id: int, grid_size: int = 400):
    scores = np.asarray(unknown_score)
    thresholds = np.linspace(scores.min(), scores.max(), grid_size)
    best = {"threshold": float(thresholds[0]), "macro_f1": -1.0}
    for threshold in thresholds:
        pred = predict_open_world_from_threshold(known_pred, scores, threshold, unknown_id)
        macro_f1 = f1_score(y_open_true, pred, average="macro", zero_division=0)
        if macro_f1 > best["macro_f1"]:
            best = {"threshold": float(threshold), "macro_f1": float(macro_f1)}
    return best


def compute_oscr(y_true_known_unknown, y_true_known_labels, known_pred, known_confidence):
    """Compute a compact OSCR approximation.

    y_true_known_unknown: boolean array, True for known samples and False for unknown.
    y_true_known_labels: known class IDs; ignored for unknown rows.
    known_pred: closed-set predicted known class IDs.
    known_confidence: higher means more likely known.
    """
    y_true_known_unknown = np.asarray(y_true_known_unknown).astype(bool)
    y_true_known_labels = np.asarray(y_true_known_labels)
    known_pred = np.asarray(known_pred)
    known_confidence = np.asarray(known_confidence)
    thresholds = np.sort(np.unique(known_confidence))[::-1]
    ccr, fpr = [], []
    n_known = max(y_true_known_unknown.sum(), 1)
    n_unknown = max((~y_true_known_unknown).sum(), 1)
    for thr in thresholds:
        accept = known_confidence >= thr
        correct_known = accept & y_true_known_unknown & (known_pred == y_true_known_labels)
        false_known = accept & (~y_true_known_unknown)
        ccr.append(correct_known.sum() / n_known)
        fpr.append(false_known.sum() / n_unknown)
    if len(fpr) < 2:
        return np.nan
    order = np.argsort(fpr)
    return float(np.trapz(np.asarray(ccr)[order], np.asarray(fpr)[order]))


def compute_open_world_metrics(y_open_true, y_open_pred, unknown_score, unknown_id: int):
    y_open_true = np.asarray(y_open_true)
    y_open_pred = np.asarray(y_open_pred)
    is_unknown = (y_open_true == unknown_id).astype(int)
    metrics = {
        "open_accuracy": float(accuracy_score(y_open_true, y_open_pred)),
        "open_macro_f1": float(f1_score(y_open_true, y_open_pred, average="macro", zero_division=0)),
        "unknown_recall": float(recall_score(is_unknown, (y_open_pred == unknown_id).astype(int), zero_division=0)),
    }
    if len(np.unique(is_unknown)) == 2:
        metrics["unknown_auroc"] = float(roc_auc_score(is_unknown, unknown_score))
    else:
        metrics["unknown_auroc"] = np.nan
    return metrics
