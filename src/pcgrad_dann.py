"""
Asymmetric PCGrad ("gradient surgery") for the DANN training step.

WHY THIS EXISTS
---------------
Standard DANN sums the classification loss and the (GRL-reversed) domain loss
and calls a single ``.backward()``::

    loss = class_loss + domain_loss_weight * domain_loss
    loss.backward()

The moment those two losses are summed, the classification gradient ``g_cls``
and the domain-alignment gradient ``g_dom`` are fused inside each parameter's
``.grad`` buffer and can no longer be pulled apart. When ``g_dom`` points
against ``g_cls`` and is strong enough to out-vote it, i.e.

    lambda_dom * |<g_dom, g_cls>|  >  ||g_cls||^2      with  <g_dom, g_cls> < 0,

the classification loss actually *increases* on that step. That is the
negative-transfer / "-0.273 Macro-F1 collapse" failure documented for the
ResNet50 + concat + full-text / MCR-SL configuration.

WHAT SURGERY DOES
-----------------
It computes the two gradients *separately*, removes the component of the domain
gradient that lies along (and fights) the classifier gradient, and only then
combines them::

    g_dom_surgery = g_dom - (<g_dom, g_cls> / ||g_cls||^2) * g_cls      # if they conflict

The leftover is orthogonal to ``g_cls`` by construction, so the domain term's
contribution along the classifier direction becomes exactly 0. To first order
this guarantees the domain objective can no longer increase the classification
loss, while domain alignment still proceeds in the (typically vast) subspace
the classifier does not care about.

ASYMMETRIC BY DESIGN
--------------------
Classification is the real goal; domain alignment is auxiliary/adversarial.
So this projects *only* the domain gradient off the classifier gradient and
never modifies the classifier gradient itself (``primary_index=0``). Standard
symmetric PCGrad -- which would also let the domain task cut into the
classifier's own gradient -- is available with ``primary_index=None``.

The Gradient Reversal Layer in ``models.py`` is orthogonal to this and stays
exactly as-is: GRL flips the *sign* of the domain gradient (making the game
adversarial); surgery fixes its *direction* relative to the classifier. They do
different jobs and compose cleanly.
"""

from __future__ import annotations

from typing import Sequence

import torch


def _flat_task_grad(loss, params, retain_graph: bool):
    """Gradient of a single loss w.r.t. params, flattened into one vector.

    Params untouched by this loss (e.g. the domain head w.r.t. class_loss)
    return None from autograd and are treated as zeros, so the flattened
    layout stays aligned across tasks.
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


def pcgrad_backward(
    losses: Sequence[torch.Tensor],
    params,
    primary_index: int | None = 0,
    eps: float = 1e-12,
) -> dict:
    """Deconflict several task gradients, then write the result into ``.grad``.

    Parameters
    ----------
    losses:
        One scalar loss per task. For DANN this is
        ``[class_loss, domain_loss_weight * domain_loss]``. Pass the losses
        themselves (NOT their sum) -- keeping them separate is the whole point.
    params:
        Iterable of model parameters. Frozen params (``requires_grad=False``)
        are ignored.
    primary_index:
        Index of the protected objective (its gradient is never modified, and
        every other gradient is projected off it). ``0`` protects
        classification. ``None`` runs symmetric PCGrad (every gradient projected
        off every conflicting gradient).
    eps:
        Numerical floor for the projection denominator.

    Returns
    -------
    dict with ``pcgrad_conflicts`` (number of conflicting pairs surgically
    corrected this step) for optional logging.

    After calling this, run ``optimizer.step()`` as usual. Do not call
    ``loss.backward()`` -- this function has already populated ``.grad``.
    """
    params = [p for p in params if p.requires_grad]
    n = len(losses)

    # 1) Each task's gradient ON ITS OWN -- what the summed backward destroyed.
    #    The last grad frees the graph (retain_graph=False); earlier ones keep it.
    flat_grads = [
        _flat_task_grad(loss, params, retain_graph=(i < n - 1))
        for i, loss in enumerate(losses)
    ]

    # 2) Detect + surgically remove the conflicting component.
    proj = [g.clone() for g in flat_grads]
    n_conflicts = 0
    for i in range(n):
        if primary_index is not None and i == primary_index:
            continue  # never modify the protected (classification) gradient
        for j in range(n):
            if i == j:
                continue
            if primary_index is not None and j != primary_index:
                continue  # asymmetric: only ever project off the primary gradient
            dot = torch.dot(proj[i], flat_grads[j])
            if dot < 0:  # the two gradients fight
                n_conflicts += 1
                proj[i] = proj[i] - (dot / (flat_grads[j].pow(2).sum() + eps)) * flat_grads[j]

    # 3) Now it is safe to add them.
    merged = torch.stack(proj).sum(dim=0)

    # 4) Write the deconflicted gradient back so optimizer.step() uses it.
    idx = 0
    for p in params:
        k = p.numel()
        p.grad = merged[idx:idx + k].view_as(p).clone()
        idx += k

    return {"pcgrad_conflicts": n_conflicts}


def dann_pcgrad_step(
    class_loss: torch.Tensor,
    weighted_domain_loss: torch.Tensor,
    model,
    eps: float = 1e-12,
) -> dict:
    """Convenience wrapper for the two-loss DANN step.

    Pass the classification loss and the *already-weighted* domain loss
    (``domain_loss_weight * domain_loss``). Classification is protected
    (``primary_index=0``). Call ``optimizer.step()`` afterwards.
    """
    return pcgrad_backward(
        losses=[class_loss, weighted_domain_loss],
        params=list(model.parameters()),
        primary_index=0,
        eps=eps,
    )
