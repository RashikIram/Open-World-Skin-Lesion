"""
Asymmetric PCGrad ("gradient surgery") for the DANN training step -- v2, CORRECTED.

WHAT CHANGED VS v1 (and why the v1 results were mixed)
-----------------------------------------------------
v1 flattened *every* model parameter into one global vector and projected the
whole domain gradient off the whole classification gradient::

    g_dom' = g_dom - (<g_dom, g_cls> / ||g_cls||^2) * g_cls        # global vector

That is the wrong place to operate. A DANN has three disjoint parameter groups:

    shared      image_encoder / text_encoder / fusion   <- BOTH losses touch these
    cls head    classifier                              <- ONLY class_loss touches
    dom head    domain_classifier                       <- ONLY domain_loss touches

On the classifier head, ``g_dom`` is exactly zero (the domain loss never depends
on that head). But the *global* projection subtracts ``alpha * g_cls`` across all
coordinates, so on the classifier-head slice it produces

    g_dom'[cls_head] = 0 - alpha * g_cls[cls_head]  != 0

i.e. surgery *manufactures* a fake domain gradient on the classification head.
After ``g_final = g_cls + g_dom'``, the classifier head is updated with
``(1 - alpha) * g_cls`` -- a silently rescaled classification step. The v1
docstring claimed the classification gradient is "never modified"; the primary
*vector* was indeed untouched, but the merged *update* on classification-only
parameters was not. The same artefact rescales the domain head.

This is measurable: on conflicting steps the classifier-head update in v1 differs
from the true ``g_cls`` by a non-zero amount. That perturbation is unmotivated
noise, it is applied only on conflict steps, and it is a plausible driver of the
mixed v1 outcome (46 wins / 61 losses / 1 tie over 108 paired seed runs; mean
PCGrad-minus-standard target Macro-F1 = -0.0126, one-sample t-test p = 0.011 --
i.e. v1 was significantly *worse* than standard DANN).

THE CORRECTION
--------------
The genuine conflict between "classify lesions" and "fool the domain critic"
lives only where both objectives write to the same weights: the shared
representation. So the projection is now restricted to the shared parameters::

    g_d' = g_d - (<g_d, g_c> / (||g_c||^2 + eps)) * g_c     over SHARED params,
                                                            and only if <g_d,g_c> < 0

    g[shared]   = g_c + g_d'
    g[cls head] = g_c            (exactly, always -- no contamination)
    g[dom head] = g_d            (exactly, always -- no contamination)

The domain adversary may still reshape the representation, but it can no longer
manufacture or rescale gradients on either task-specific head. Dot products,
norms and the conflict test are all computed on the shared slice alone, so the
conflict statistic now measures the quantity it is supposed to measure.

GRL INTERACTION (unchanged, and deliberately so)
------------------------------------------------
The Gradient Reversal Layer in the model flips the *sign* of the domain gradient,
which is what makes the game adversarial. Surgery then fixes that gradient's
*direction* relative to the classifier. Pass the post-GRL losses exactly as
before; the two mechanisms compose and neither is double-applied here.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


# Root module names that carry the shared representation in this project's DANN.
# `ExactDANN` in the runner preserves exactly these names, as does models.py.
DEFAULT_SHARED_PREFIXES: tuple[str, ...] = (
    "image_encoder",
    "text_encoder",
    "fusion",
)


def _flat_grad(loss, params, retain_graph: bool) -> torch.Tensor:
    """Gradient of one loss w.r.t. ``params``, flattened into a single vector.

    Parameters this loss does not touch return None from autograd and are
    treated as zeros, so the flat layout stays aligned across losses.
    """
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, params)
        ]
    )


def split_shared_params(
    model,
    shared_prefixes: Sequence[str] = DEFAULT_SHARED_PREFIXES,
) -> tuple[list, list[bool]]:
    """Return trainable params plus a per-param mask marking the shared ones.

    A parameter is "shared" when its qualified name starts with one of
    ``shared_prefixes`` -- i.e. it lives in the representation that both the
    classifier and the domain critic read from.
    """
    params, is_shared = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        params.append(p)
        is_shared.append(any(name.startswith(pref) for pref in shared_prefixes))

    if not any(is_shared):
        raise ValueError(
            "No shared parameters matched "
            f"{tuple(shared_prefixes)}. Surgery would be a no-op. "
            "Check the model's module names via model.named_parameters()."
        )
    return params, is_shared


def _shared_index_mask(params, is_shared: Sequence[bool]) -> torch.Tensor:
    """Boolean mask over the flattened parameter vector, True on shared coords."""
    chunks = [
        torch.full((p.numel(),), bool(flag), dtype=torch.bool, device=p.device)
        for p, flag in zip(params, is_shared)
    ]
    return torch.cat(chunks)


def pcgrad_backward_shared(
    class_loss: torch.Tensor,
    weighted_domain_loss: torch.Tensor,
    params: Iterable,
    is_shared: Sequence[bool],
    eps: float = 1e-12,
) -> dict:
    """Asymmetric PCGrad restricted to the shared representation.

    Computes ``g_cls`` and ``g_dom`` separately, projects ``g_dom`` off ``g_cls``
    **on the shared coordinates only** and **only when they conflict**, then
    writes the merged gradient into ``.grad``. Task-specific heads receive their
    own gradient untouched.

    Parameters
    ----------
    class_loss:
        Scalar classification loss (the protected objective).
    weighted_domain_loss:
        Scalar, *already weighted* domain loss (``lambda_dom * L_dom``), taken
        after the GRL as usual.
    params:
        Trainable parameters, in the same order as ``is_shared``.
    is_shared:
        Per-parameter flag from :func:`split_shared_params`.
    eps:
        Numerical floor for the projection denominator.

    Returns
    -------
    dict with diagnostics:
        ``pcgrad_conflicts``        1 if this step conflicted, else 0
        ``pcgrad_cosine``           cosine(g_dom, g_cls) on the shared slice
        ``pcgrad_dot``              <g_dom, g_cls> on the shared slice
        ``pcgrad_projection_scale`` alpha actually subtracted (0.0 if no conflict)

    After calling this, run ``optimizer.step()``. Do **not** call
    ``loss.backward()`` -- ``.grad`` is already populated.
    """
    params = list(params)
    if len(params) != len(is_shared):
        raise ValueError(
            f"params ({len(params)}) and is_shared ({len(is_shared)}) "
            "must describe the same parameters in the same order."
        )

    # 1) The two gradients ON THEIR OWN -- what a summed backward() destroys.
    g_cls = _flat_grad(class_loss, params, retain_graph=True)
    g_dom = _flat_grad(weighted_domain_loss, params, retain_graph=False)

    mask = _shared_index_mask(params, is_shared)

    # 2) Conflict is judged ONLY on the shared representation, because that is
    #    the only place where the two objectives actually compete for weights.
    gc_s = g_cls[mask]
    gd_s = g_dom[mask]

    dot = torch.dot(gd_s, gc_s)
    gc_sq = gc_s.pow(2).sum()

    n_conflicts = 0
    alpha = torch.zeros((), dtype=g_cls.dtype, device=g_cls.device)
    if bool(dot < 0):
        n_conflicts = 1
        alpha = dot / (gc_sq + eps)

    # 3) Merge. Start from the honest sum, then correct the shared slice only.
    merged = g_cls + g_dom
    if n_conflicts:
        # g_dom'[shared] = g_dom[shared] - alpha * g_cls[shared]
        merged[mask] = gc_s + (gd_s - alpha * gc_s)

    # Heads are left exactly as their own loss produced them: on the classifier
    # head g_dom is identically zero, so merged == g_cls there, and vice versa.

    cos = dot / (gc_s.norm() * gd_s.norm() + eps)

    # 4) Write back so optimizer.step() consumes the deconflicted gradient.
    idx = 0
    for p in params:
        k = p.numel()
        p.grad = merged[idx:idx + k].view_as(p).clone()
        idx += k

    return {
        "pcgrad_conflicts": n_conflicts,
        "pcgrad_cosine": float(cos),
        "pcgrad_dot": float(dot),
        "pcgrad_projection_scale": float(alpha),
    }


def dann_pcgrad_step(
    class_loss: torch.Tensor,
    weighted_domain_loss: torch.Tensor,
    model,
    eps: float = 1e-12,
    shared_prefixes: Sequence[str] = DEFAULT_SHARED_PREFIXES,
) -> dict:
    """Drop-in replacement for the v1 wrapper, now shared-parameter-scoped.

    Signature is backward compatible with v1 for the first four arguments, so
    existing call sites (``dann_pcgrad_step(class_loss, weighted_domain_loss,
    model)``) keep working unchanged. Call ``optimizer.step()`` afterwards.
    """
    params, is_shared = split_shared_params(model, shared_prefixes)
    return pcgrad_backward_shared(
        class_loss=class_loss,
        weighted_domain_loss=weighted_domain_loss,
        params=params,
        is_shared=is_shared,
        eps=eps,
    )
