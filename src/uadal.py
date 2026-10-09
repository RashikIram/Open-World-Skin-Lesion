"""UADAL building blocks ported from the official implementation.

Jang et al., "Unknown-Aware Domain Adversarial Learning for Open-Set Domain
Adaptation", NeurIPS 2022. https://github.com/JoonHo-Jang/UADAL

Ported pieces (original location in the UADAL repo):
- BetaMixture1D                  -> models/function.py (BetaMixture1D, fit_beta_weighted)
- soft_cross_entropy             -> models/function.py (CrossEntropyLoss)
- prediction_entropy             -> models/function.py (HLoss)
- smoothed_one_hot               -> models/model_UADAL.py (label smoothing blocks)
- normalized entropy + posterior -> models/model_UADAL.py (test / compute_probabilities_batch)

The training loop itself lives in src/train_uadal.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F


# ============================================================
# Losses
# ============================================================

def smoothed_one_hot(labels: torch.Tensor, num_classes: int, eps: float) -> torch.Tensor:
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    return one_hot * (1.0 - eps) + eps / num_classes


def soft_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    instance_weight: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Cross-entropy against a soft (possibly signed) target distribution.

    Matches UADAL's CrossEntropyLoss: with instance weights, the loss is the
    weighted sum divided by the weight sum rather than by the batch size.
    """
    probs = F.softmax(logits, dim=1)
    ce = -target * torch.log(probs + eps)

    if instance_weight is None:
        return ce.sum() / float(logits.size(0))

    weight = instance_weight.view(-1, 1)
    return (weight * ce).sum() / (weight.sum() + eps)


def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)


# ============================================================
# Beta mixture posterior over normalized open-set entropy
# ============================================================

def _fit_beta_weighted(x: np.ndarray, w: np.ndarray):
    x_bar = np.sum(w * x) / np.sum(w)
    s2 = np.sum(w * (x - x_bar) ** 2) / np.sum(w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) / x_bar
    return alpha, beta


class BetaMixture1D:
    """Two-component beta mixture.

    Component 0 starts near 0 (low entropy, known) and component 1 near 1
    (high entropy, unknown), so posterior(x, 1) is p(unknown | entropy).
    """

    def __init__(self, max_iters: int = 10):
        self.alphas = np.array([1.0, 2.0], dtype=np.float64)
        self.betas = np.array([2.0, 1.0], dtype=np.float64)
        self.weight = np.array([0.5, 0.5], dtype=np.float64)
        self.max_iters = max_iters
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = np.array([self.weighted_likelihood(x, i) for i in range(2)])
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def fit(self, x: np.ndarray):
        x = np.clip(np.copy(x), 1e-4, 1 - 1e-4)
        for _ in range(self.max_iters):
            r = self.responsibilities(x)
            self.alphas[0], self.betas[0] = _fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = _fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()
        return self

    def is_valid(self) -> bool:
        params = np.concatenate([self.alphas, self.betas, self.weight])
        return bool(np.all(np.isfinite(params)) and np.all(self.alphas > 0) and np.all(self.betas > 0))

    def summary(self) -> dict:
        return {
            "bmm_weight_known": float(self.weight[0]),
            "bmm_weight_unknown": float(self.weight[1]),
            "bmm_alpha_known": float(self.alphas[0]),
            "bmm_beta_known": float(self.betas[0]),
            "bmm_alpha_unknown": float(self.alphas[1]),
            "bmm_beta_unknown": float(self.betas[1]),
        }


def fit_unknown_posterior(entropy: np.ndarray, num_known_classes: int):
    """Fit the BMM on entropy / log(K) and return (p_unknown, bmm, valid).

    If the fit degenerates (possible when unknowns are very rare), every
    sample falls back to p_unknown = 0, i.e. plain known-class alignment.
    """
    normalized = np.asarray(entropy, dtype=np.float64) / math.log(num_known_classes)
    fit_input = np.clip(normalized, 1e-3, 1 - 1e-3)
    bmm = BetaMixture1D().fit(fit_input)

    if not bmm.is_valid():
        return np.zeros_like(normalized, dtype=np.float32), bmm, False

    posterior = bmm.posterior(np.clip(normalized, 1e-4, 1 - 1e-4), 1)
    posterior = np.nan_to_num(posterior, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(posterior, 0.0, 1.0).astype(np.float32), bmm, True


# ============================================================
# Checkpoint loading
# ============================================================

def load_source_checkpoint_into_uadal(model: torch.nn.Module, checkpoint_path: str | Path) -> dict:
    """Load a closed-set (K-way) or DANN checkpoint into a UADAL model.

    Matching tensors are copied as-is. The K-way final classifier layer is
    copied into the first K rows of the (K+1)-way UADAL classifier, so the
    unknown row is the only newly initialised output.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("model_state_dict", checkpoint)
    target_state = model.state_dict()

    new_state, expanded, skipped = {}, [], []
    for key, value in source_state.items():
        if key not in target_state:
            skipped.append(key)
            continue

        target_value = target_state[key]
        if value.shape == target_value.shape:
            new_state[key] = value
        elif (
            key.startswith("classifier.")
            and value.shape[0] + 1 == target_value.shape[0]
            and value.shape[1:] == target_value.shape[1:]
        ):
            merged = target_value.clone()
            merged[: value.shape[0]] = value
            new_state[key] = merged
            expanded.append(key)
        else:
            skipped.append(key)

    missing, _ = model.load_state_dict(new_state, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "n_loaded_tensors": len(new_state),
        "expanded_to_k_plus_1": expanded,
        "skipped_keys": skipped,
        "missing_keys": list(missing),
    }
