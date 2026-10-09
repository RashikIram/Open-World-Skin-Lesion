"""UADAL open-set domain adaptation for the image-text fusion models.

Port of Jang et al., "Unknown-Aware Domain Adversarial Learning for Open-Set
Domain Adaptation" (NeurIPS 2022), https://github.com/JoonHo-Jang/UADAL
(models/model_UADAL.py: train_init, train, test), adapted to this pipeline's
loaders, checkpoints and evaluation outputs.

Differences from DANN (src/train_dann.py):
- the unlabeled target adaptation set keeps its unknown-class images;
  target labels are never used for training
- the classifier has K+1 outputs; index K is "unknown"
- a 3-way discriminator separates source / target-known / target-unknown,
  weighted per sample by a beta-mixture posterior p(unknown | entropy)
- open-world prediction needs no threshold: argmax == K means unknown
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import ExperimentConfig, KNOWN_CLASSES, SEED, UNKNOWN_LABEL_NAME
from .datasets import DomainAdaptationDataset, TwoViewDomainAdaptationDataset
from .metrics import (
    compute_open_world_metrics,
    compute_os_star_hos,
    evaluate_closed_set,
    find_best_unknown_threshold,
    predict_open_world_from_threshold,
)
from .models import build_uadal_model
from .preprocessing import add_known_unknown_columns, load_standardized_splits
from .train_dann import (
    build_sampler,
    dann_lambda_schedule,
    evaluate_known_split,
    label_name,
    make_loader,
    metric_value,
    model_stem,
    sanitize_name,
    save_json_safe,
    save_open_world_artifacts,
)
from .transforms import (
    get_eval_transform,
    get_strong_target_transform,
    get_train_transform,
    get_weak_train_transform,
)
from .uadal import (
    fit_unknown_posterior,
    load_source_checkpoint_into_uadal,
    prediction_entropy,
    smoothed_one_hot,
    soft_cross_entropy,
)
from .utils import ensure_dir, get_device, save_checkpoint, seed_everything


DOMAIN_SOURCE, DOMAIN_TARGET_KNOWN, DOMAIN_TARGET_UNKNOWN = 0, 1, 2


# ============================================================
# Optimizers
# ============================================================

def uadal_param_groups(model, base_lr: float, head_lr_mult: float):
    """Backbone/fusion (G) at base_lr; C and E heads at base_lr * head_lr_mult."""
    head_prefixes = ("classifier.", "open_set_recognizer.")
    backbone, heads = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad or name.startswith("uadal_domain_classifier."):
            continue
        (heads if name.startswith(head_prefixes) else backbone).append(param)
    return [
        {"params": backbone, "lr": base_lr},
        {"params": heads, "lr": base_lr * head_lr_mult},
    ]


def cosine_lambda(total_steps: int):
    total_steps = max(total_steps, 1)
    return lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps))


def reset_open_set_recognizer(model, optimizer):
    """UADAL re-initialises E after every posterior refit; drop its stale Adam state too."""
    model.reset_open_set_recognizer()
    for p in model.open_set_recognizer.parameters():
        optimizer.state.pop(p, None)


def source_losses(model, feat_s, label_s, num_known: int, ls_eps: float):
    loss_e = soft_cross_entropy(
        model.open_set_recognizer(feat_s),
        smoothed_one_hot(label_s, num_known, ls_eps),
    )
    loss_c = soft_cross_entropy(
        model.classifier(feat_s),
        smoothed_one_hot(label_s, num_known + 1, ls_eps),
    )
    return loss_e, loss_c


def features(model, batch, pixel_key: str = "pixel_values"):
    return model.extract_features(batch[pixel_key], batch.get("input_ids"), batch.get("attention_mask"))


# ============================================================
# Stage 1: source warm-up (UADAL train_init)
# ============================================================

def warmup_source(model, source_loader, device, args, cfg):
    if len(source_loader) == 0:
        raise ValueError("UADAL warm-up requires a non-empty source loader (check --batch-size vs. source size).")
    num_known = model.num_known_classes
    optimizer = torch.optim.AdamW(
        uadal_param_groups(model, cfg.lr, args.head_lr_mult),
        weight_decay=cfg.weight_decay,
    )
    model.train()

    step = 0
    progress = tqdm(total=args.warmup_steps, desc="uadal warmup", leave=False)
    while step < args.warmup_steps:
        for src in source_loader:
            if step >= args.warmup_steps:
                break
            src = {k: v.to(device) if hasattr(v, "to") else v for k, v in src.items()}

            optimizer.zero_grad(set_to_none=True)
            feat_s = features(model, src)
            loss_e, loss_c = source_losses(model, feat_s, src["label"], num_known, args.ls_eps)
            (loss_e + loss_c).backward()
            optimizer.step()

            step += 1
            progress.update(1)
    progress.close()


# ============================================================
# Posterior inference (UADAL test() refit + compute_probabilities_batch)
# ============================================================

@torch.no_grad()
def refit_unknown_posterior(model, posterior_loader, device, n_target: int, is_unknown_diag=None):
    """
    Fit the BMM on E's entropy over the whole target adaptation set.

    UADAL keeps frozen copies of G and E and recomputes this posterior per batch
    on the un-augmented image. Those copies never change between refits, so
    precomputing it once per refit is equivalent and avoids duplicating the
    backbone + text encoder in GPU memory.
    """
    model.eval()
    entropy = np.zeros(n_target, dtype=np.float64)

    for batch in tqdm(posterior_loader, desc="uadal posterior", leave=False):
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        ent = prediction_entropy(model.open_set_recognizer(features(model, batch)))
        entropy[batch["index"].cpu().numpy()] = ent.cpu().numpy()

    w_unk, bmm, valid = fit_unknown_posterior(entropy, model.num_known_classes)

    stats = {
        "bmm_valid": valid,
        **bmm.summary(),
        "w_unk_mean": float(w_unk.mean()),
        "w_unk_frac_gt_0_5": float((w_unk > 0.5).mean()),
    }

    # Logged only for monitoring; target labels never enter the training loss.
    if is_unknown_diag is not None and len(np.unique(is_unknown_diag)) == 2:
        stats["diag_true_unknown_frac"] = float(np.mean(is_unknown_diag))
        stats["diag_w_unk_auroc"] = float(roc_auc_score(is_unknown_diag, w_unk))

    model.train()
    return torch.tensor(w_unk, dtype=torch.float32, device=device), stats


# ============================================================
# Stage 2: unknown-aware adversarial adaptation (UADAL train)
# ============================================================

def train_uadal_epoch(
    *,
    model,
    source_loader,
    target_loader,
    w_unk_all,
    optimizer,
    scheduler,
    disc_optimizer,
    disc_scheduler,
    device,
    epoch: int,
    total_epochs: int,
    args,
):
    model.train()
    disc = model.uadal_domain_classifier
    num_known = model.num_known_classes

    n_steps = min(len(source_loader), len(target_loader))
    if n_steps <= 0:
        raise ValueError(
            "UADAL training requires non-empty source and target loaders. "
            f"Got len(source_loader)={len(source_loader)}, len(target_loader)={len(target_loader)}."
        )

    totals = {k: 0.0 for k in ["loss", "loss_d", "loss_g", "loss_src_e", "loss_src_c", "loss_tgt_unknown", "loss_tgt_pseudo", "pseudo_mask_rate"]}

    for local_step, (src, tgt) in enumerate(
        tqdm(zip(source_loader, target_loader), total=n_steps, desc="uadal", leave=False)
    ):
        if local_step >= n_steps:
            break

        global_step = (epoch - 1) * n_steps + local_step
        alpha = dann_lambda_schedule(global_step, n_steps * total_epochs, 1.0)

        src = {k: v.to(device) if hasattr(v, "to") else v for k, v in src.items()}
        tgt = {k: v.to(device) if hasattr(v, "to") else v for k, v in tgt.items()}

        w_unk = w_unk_all[tgt["index"]]
        w_k = 1.0 - w_unk

        feat_s = features(model, src)
        feat_t = features(model, tgt)
        feat_t_aug = features(model, tgt, "pixel_values_strong")

        n_s, n_t = feat_s.size(0), feat_t.size(0)
        label_ds = F.one_hot(torch.full((n_s,), DOMAIN_SOURCE, device=device), 3).float()
        label_dt_known = F.one_hot(torch.full((n_t,), DOMAIN_TARGET_KNOWN, device=device), 3).float()
        label_dt_unknown = F.one_hot(torch.full((n_t,), DOMAIN_TARGET_UNKNOWN, device=device), 3).float()

        # ---- D step: classify source / target-known / target-unknown ----
        # G is not updated between this step and the G step, so the same
        # features (detached here) are reused instead of a second forward pass.
        disc_optimizer.zero_grad(set_to_none=True)
        target_dt = w_k[:, None] * label_dt_known + w_unk[:, None] * label_dt_unknown
        loss_d = 0.5 * (
            soft_cross_entropy(disc(feat_s.detach()), label_ds)
            + soft_cross_entropy(disc(feat_t.detach()), target_dt)
        )
        loss_d.backward()
        if args.disc_clip > 0:
            torch.nn.utils.clip_grad_norm_(disc.parameters(), args.disc_clip)
        disc_optimizer.step()
        disc_scheduler.step()

        # ---- G / C / E step ----
        optimizer.zero_grad(set_to_none=True)

        # Signed target: pull target-known towards source, push target-unknown away.
        loss_ds = soft_cross_entropy(disc(feat_s), label_ds)
        loss_dt = soft_cross_entropy(
            disc(feat_t),
            w_k[:, None] * label_dt_known - w_unk[:, None] * label_dt_unknown,
        )
        loss_g = alpha * (-loss_ds - loss_dt)

        loss_src_e, loss_src_c = source_losses(model, feat_s, src["label"], num_known, args.ls_eps)

        out_ct = model.classifier(feat_t)
        out_ct_aug = model.classifier(feat_t_aug)

        unknown_target = smoothed_one_hot(
            torch.full((n_t,), num_known, device=device), num_known + 1, args.ls_eps
        )
        loss_tgt_unknown = alpha * soft_cross_entropy(out_ct_aug, unknown_target, instance_weight=w_unk)

        pseudo = torch.softmax(out_ct.detach(), dim=-1)
        max_probs, pseudo_labels = pseudo.max(dim=-1)
        mask = max_probs.ge(args.pseudo_threshold).float()
        loss_tgt_pseudo = soft_cross_entropy(
            out_ct_aug,
            F.one_hot(pseudo_labels, num_known + 1).float(),
            instance_weight=mask,
        )

        loss = (
            loss_src_e
            + loss_src_c
            + args.adv_weight * loss_g
            + args.pseudo_weight * loss_tgt_pseudo
            + args.unknown_weight * loss_tgt_unknown
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        # loss_g also left gradients on D; they are cleared by the next D step.

        totals["loss"] += float(loss.item())
        totals["loss_d"] += float(loss_d.item())
        totals["loss_g"] += float(loss_g.item())
        totals["loss_src_e"] += float(loss_src_e.item())
        totals["loss_src_c"] += float(loss_src_c.item())
        totals["loss_tgt_unknown"] += float(loss_tgt_unknown.item())
        totals["loss_tgt_pseudo"] += float(loss_tgt_pseudo.item())
        totals["pseudo_mask_rate"] += float(mask.mean().item())

    return {**{k: v / n_steps for k, v in totals.items()}, "n_steps": n_steps}


# ============================================================
# Open-world evaluation
# ============================================================

@torch.no_grad()
def collect_uadal_outputs(model, loader, device):
    model.eval()
    open_logits_all, labels_all, indices_all = [], [], []

    for batch in tqdm(loader, desc="collect", leave=False):
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        _, open_logits, _ = model(batch["pixel_values"], batch.get("input_ids"), batch.get("attention_mask"))
        open_logits_all.append(open_logits.cpu().numpy())
        labels_all.extend(batch["label"].cpu().numpy().tolist())
        indices_all.extend(batch["index"].cpu().numpy().tolist())

    if not open_logits_all:
        raise ValueError("Cannot evaluate open-world split because the loader is empty.")

    return np.concatenate(open_logits_all, axis=0), np.asarray(labels_all), np.asarray(indices_all)


def evaluate_uadal_open_world_split(
    *,
    model,
    df: pd.DataFrame,
    tokenizer,
    text_col: str,
    batch_size: int,
    device,
    cfg: ExperimentConfig,
    threshold: float | None = None,
    rule: str = "native",
):
    """
    rule="native": UADAL's own decision, unknown when argmax over K+1 is K.
    rule="tuned":  unknown when p(unknown) >= threshold; if threshold is None
                   it is chosen on this split (use only on target_val).
    """
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
    open_logits, y_open, indices = collect_uadal_outputs(model, loader, device)

    unknown_id = len(KNOWN_CLASSES)
    open_probs = torch.softmax(torch.tensor(open_logits), dim=1).numpy()
    known_probs = torch.softmax(torch.tensor(open_logits[:, :unknown_id]), dim=1).numpy()
    known_pred = known_probs.argmax(axis=1)
    unknown_score = open_probs[:, unknown_id]

    if rule == "native":
        y_open_pred = open_probs.argmax(axis=1)
        threshold_rule = "predict UNKNOWN when argmax over K+1 outputs is the unknown class"
    else:
        if threshold is None:
            threshold = float(find_best_unknown_threshold(y_open, known_pred, unknown_score, unknown_id)["threshold"])
        y_open_pred = predict_open_world_from_threshold(known_pred, unknown_score, threshold, unknown_id)
        threshold_rule = "predict UNKNOWN when p_unknown >= threshold_unknown_score"

    metrics = compute_open_world_metrics(
        y_open,
        y_open_pred,
        unknown_score,
        unknown_id,
        known_pred=known_pred,
        known_confidence=-unknown_score,
    )
    metrics.update(compute_os_star_hos(y_open, y_open_pred, unknown_id))
    metrics["scoring_method"] = "uadal_p_unknown"
    metrics["decision_rule"] = rule
    metrics["threshold_rule"] = threshold_rule
    metrics["threshold_unknown_score"] = float(threshold) if threshold is not None else float("nan")

    pred_df = df.iloc[indices].copy().reset_index(drop=True)
    class_names = list(KNOWN_CLASSES) + [UNKNOWN_LABEL_NAME]

    pred_df["y_open_true"] = y_open
    pred_df["y_open_true_label"] = [label_name(x, class_names) for x in y_open]
    pred_df["known_pred"] = known_pred
    pred_df["known_pred_label"] = [label_name(x, list(KNOWN_CLASSES)) for x in known_pred]
    pred_df["known_confidence"] = -unknown_score
    pred_df["unknown_score"] = unknown_score
    pred_df["y_open_pred"] = y_open_pred
    pred_df["y_open_pred_label"] = [label_name(x, class_names) for x in y_open_pred]

    for i, name in enumerate(class_names):
        pred_df[f"logit_{name}"] = open_logits[:, i]
        pred_df[f"prob_{name}"] = open_probs[:, i]

    return metrics, pred_df


# ============================================================
# Main run
# ============================================================

def run(args):
    seed_everything(args.seed)

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

    source_df = add_known_unknown_columns(
        load_standardized_splits(args.standardized_csv, args.source_image_roots, args.source_dataset)
    )
    target_df = add_known_unknown_columns(
        load_standardized_splits(args.standardized_csv, args.target_image_roots, args.target_dataset)
    )

    source_train_df = source_df[
        (source_df["split"].isin(["source_train", "train"]))
        & (~source_df["is_unknown"])
    ].copy()

    # Open-set adaptation: unknown target images stay in the unlabeled adaptation set.
    target_adapt_df = target_df[target_df["split"] == "target_adapt"].copy().reset_index(drop=True)

    target_val_df = target_df[target_df["split"] == "target_val"].copy()
    target_test_df = target_df[target_df["split"] == "target_test"].copy()
    target_val_known_df = target_val_df[~target_val_df["is_unknown"]].copy()
    target_test_known_df = target_test_df[~target_test_df["is_unknown"]].copy()

    if source_train_df.empty:
        raise ValueError("No known-class source training rows found.")
    if target_adapt_df.empty:
        raise ValueError("No target adaptation rows found.")
    if target_val_known_df.empty:
        raise ValueError("No known-class target validation rows found.")

    split_sizes = {
        "source_train_known": int(len(source_train_df)),
        "target_adapt_all": int(len(target_adapt_df)),
        "target_adapt_unknown_diag": int(target_adapt_df["is_unknown"].sum()),
        "target_val_all": int(len(target_val_df)),
        "target_val_known": int(len(target_val_known_df)),
        "target_test_all": int(len(target_test_df)),
        "target_test_known": int(len(target_test_known_df)),
    }
    save_json_safe(split_sizes, output_dir / "split_sizes.json")
    print(split_sizes)

    source_train_df.to_csv(output_dir / "source_train_known_split.csv", index=False)
    target_adapt_df.to_csv(output_dir / "target_adapt_all_split.csv", index=False)
    target_val_df.to_csv(output_dir / "target_val_all_split.csv", index=False)
    target_test_df.to_csv(output_dir / "target_test_all_split.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    source_ds = DomainAdaptationDataset(
        source_train_df, tokenizer, get_train_transform(), args.text_col,
        domain_label=0, max_text_len=cfg.max_text_len,
    )
    target_ds = TwoViewDomainAdaptationDataset(
        target_adapt_df, tokenizer, get_weak_train_transform(), get_strong_target_transform(), args.text_col,
        label_col="label_open_id", domain_label=1, max_text_len=cfg.max_text_len,
    )
    posterior_ds = DomainAdaptationDataset(
        target_adapt_df, tokenizer, get_eval_transform(), args.text_col,
        label_col="label_open_id", domain_label=None, max_text_len=cfg.max_text_len,
    )
    val_known_ds = DomainAdaptationDataset(
        target_val_known_df, tokenizer, get_eval_transform(), args.text_col,
        domain_label=1, max_text_len=cfg.max_text_len,
    )

    # Class-balanced source as in UADAL; the target is plain shuffled because
    # its labels are not available during adaptation.
    source_loader = DataLoader(
        source_ds,
        batch_size=args.batch_size,
        sampler=build_sampler(source_train_df, args.sampler, cfg),
        shuffle=False if args.sampler != "none" else True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    target_loader = DataLoader(
        target_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    posterior_loader = make_loader(posterior_ds, args.batch_size, cfg.num_workers, sampler=None, shuffle=False)
    val_known_loader = make_loader(val_known_ds, args.batch_size, cfg.num_workers, sampler=None, shuffle=False)

    model = build_uadal_model(
        args.model_family,
        image_model_name,
        cfg.text_model_name,
        len(KNOWN_CLASSES),
        cfg.fusion_dim,
        cfg.num_heads,
        discriminator_hidden=args.disc_hidden,
    ).to(device)

    loaded_checkpoint_info = {}
    if args.checkpoint:
        loaded_checkpoint_info = load_source_checkpoint_into_uadal(model, args.checkpoint)
        save_json_safe(loaded_checkpoint_info, output_dir / "loaded_checkpoint_info.json")
        model.to(device)

    eval_criterion = nn.CrossEntropyLoss()
    is_unknown_diag = target_adapt_df["is_unknown"].to_numpy().astype(int)

    # --------------------------------------------------------
    # Before-UADAL known-only target evaluation
    # --------------------------------------------------------
    before_metrics = {}
    for split_name, split_df in [
        ("before_uadal_target_val_known", target_val_known_df),
        ("before_uadal_target_test_known", target_test_known_df),
    ]:
        m = evaluate_known_split(
            model=model, df=split_df, tokenizer=tokenizer, text_col=args.text_col,
            batch_size=args.batch_size, device=device, criterion=eval_criterion,
            cfg=cfg, output_dir=output_dir, split_name=split_name,
        )
        if m is not None:
            before_metrics[split_name] = m
    save_json_safe(before_metrics, output_dir / "before_uadal_metrics.json")

    # --------------------------------------------------------
    # Stage 1: source warm-up, then first posterior fit
    # --------------------------------------------------------
    warmup_source(model, source_loader, device, args, cfg)

    w_unk_all, posterior_stats = refit_unknown_posterior(
        model, posterior_loader, device, len(target_adapt_df), is_unknown_diag
    )
    posterior_history = [{"epoch": 0, **posterior_stats}]
    print("posterior after warm-up:", posterior_stats)

    n_steps = min(len(source_loader), len(target_loader))
    total_steps = n_steps * args.epochs

    optimizer = torch.optim.AdamW(
        uadal_param_groups(model, cfg.lr, args.head_lr_mult),
        weight_decay=cfg.weight_decay,
    )
    disc_optimizer = torch.optim.AdamW(
        model.uadal_domain_classifier.parameters(),
        lr=cfg.lr * args.head_lr_mult,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lambda(total_steps))
    disc_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_optimizer, cosine_lambda(total_steps))

    reset_open_set_recognizer(model, optimizer)

    # --------------------------------------------------------
    # Stage 2: UADAL adaptation with early stopping
    # --------------------------------------------------------
    best_score = -1.0
    best_epoch = 0
    patience_left = args.patience if args.patience is not None else cfg.patience
    history = []

    best_ckpt = output_dir / "best_uadal.pt"
    best_named_ckpt = output_dir / f"{stem}_{target_name}_{args.text_col}_best_uadal.pt"
    final_ckpt = output_dir / "final_uadal.pt"
    final_named_ckpt = output_dir / f"{stem}_{target_name}_{args.text_col}_final_uadal.pt"

    def checkpoint_extra(epoch: int, role: str) -> dict:
        return {
            "epoch": epoch,
            "model_family": args.model_family,
            "text_col": args.text_col,
            "source_dataset": args.source_dataset,
            "target_dataset": args.target_dataset,
            "known_classes": list(KNOWN_CLASSES),
            "num_outputs": len(KNOWN_CLASSES) + 1,
            "method": "UADAL",
            "selection_metric": args.selection_metric,
            "checkpoint_role": role,
        }

    for epoch in range(1, args.epochs + 1):
        train_stats = train_uadal_epoch(
            model=model,
            source_loader=source_loader,
            target_loader=target_loader,
            w_unk_all=w_unk_all,
            optimizer=optimizer,
            scheduler=scheduler,
            disc_optimizer=disc_optimizer,
            disc_scheduler=disc_scheduler,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            args=args,
        )

        val_metrics, _, _, _ = evaluate_closed_set(
            model, val_known_loader, eval_criterion, device, len(KNOWN_CLASSES)
        )
        val_f1 = metric_value(val_metrics, "macro_f1", "f1_macro")

        row = {
            "epoch": epoch,
            **train_stats,
            **{f"target_val_known_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }

        if args.selection_metric == "hos":
            val_open, _ = evaluate_uadal_open_world_split(
                model=model, df=target_val_df.reset_index(drop=True), tokenizer=tokenizer,
                text_col=args.text_col, batch_size=args.batch_size, device=device, cfg=cfg, rule="native",
            )
            row["target_val_native_hos"] = val_open["hos"]
            score = val_open["hos"] if not np.isnan(val_open["hos"]) else -1.0
        elif args.selection_metric == "last":
            score = float(epoch)
        else:
            score = val_f1

        history.append(row)
        print(row)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience_left = args.patience if args.patience is not None else cfg.patience
            extra = checkpoint_extra(epoch, "best_uadal")
            extra["best_selection_score"] = float(best_score)
            save_checkpoint(model, best_ckpt, extra)
            save_checkpoint(model, best_named_ckpt, extra)
        elif args.selection_metric != "last":
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
                break

        # UADAL refits the posterior every `update_term` epochs and resets E
        # (after checkpointing, so saved checkpoints keep a trained E).
        if epoch % args.bmm_update_every == 0 and epoch < args.epochs:
            w_unk_all, posterior_stats = refit_unknown_posterior(
                model, posterior_loader, device, len(target_adapt_df), is_unknown_diag
            )
            reset_open_set_recognizer(model, optimizer)
            posterior_history.append({"epoch": epoch, **posterior_stats})
            print(f"posterior refit after epoch {epoch}:", posterior_stats)

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame(posterior_history).to_csv(output_dir / "posterior_history.csv", index=False)

    last_epoch = history[-1]["epoch"] if history else 0
    save_checkpoint(model, final_ckpt, checkpoint_extra(last_epoch, "final_uadal"))
    save_checkpoint(model, final_named_ckpt, checkpoint_extra(last_epoch, "final_uadal"))

    # --------------------------------------------------------
    # Load best UADAL and evaluate after-UADAL known-only splits
    # --------------------------------------------------------
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    after_metrics = {}
    for split_name, split_df in [
        ("after_uadal_target_val_known", target_val_known_df),
        ("after_uadal_target_test_known", target_test_known_df),
    ]:
        m = evaluate_known_split(
            model=model, df=split_df, tokenizer=tokenizer, text_col=args.text_col,
            batch_size=args.batch_size, device=device, criterion=eval_criterion,
            cfg=cfg, output_dir=output_dir, split_name=split_name,
        )
        if m is not None:
            after_metrics[split_name] = m
    save_json_safe(after_metrics, output_dir / "after_uadal_known_metrics.json")

    # --------------------------------------------------------
    # Open-world evaluation: native K+1 rule, plus val-tuned threshold on p(unknown)
    # --------------------------------------------------------
    openworld_metrics = {}
    if not args.skip_openworld and not target_val_df.empty:
        eval_kwargs = dict(
            model=model, tokenizer=tokenizer, text_col=args.text_col,
            batch_size=args.batch_size, device=device, cfg=cfg,
        )
        val_df = target_val_df.reset_index(drop=True)
        test_df = target_test_df.reset_index(drop=True)

        runs = [("openworld_target_val_native", val_df, "native", None)]
        val_tuned, val_tuned_pred = evaluate_uadal_open_world_split(df=val_df, rule="tuned", **eval_kwargs)
        save_open_world_artifacts(output_dir=output_dir, split_name="openworld_target_val_tuned", metrics=val_tuned, pred_df=val_tuned_pred)
        openworld_metrics["openworld_target_val_tuned"] = val_tuned

        if not test_df.empty:
            runs += [
                ("openworld_target_test_native", test_df, "native", None),
                ("openworld_target_test_tuned", test_df, "tuned", val_tuned["threshold_unknown_score"]),
            ]

        for split_name, split_df, rule, threshold in runs:
            m, pred_df = evaluate_uadal_open_world_split(df=split_df, rule=rule, threshold=threshold, **eval_kwargs)
            save_open_world_artifacts(output_dir=output_dir, split_name=split_name, metrics=m, pred_df=pred_df)
            openworld_metrics[split_name] = m

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    summary = {
        "method": "UADAL",
        "model_family": args.model_family,
        "text_col": args.text_col,
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "output_dir": str(output_dir),
        "loaded_checkpoint": loaded_checkpoint_info,
        "split_sizes": split_sizes,
        "hyperparameters": {k: v for k, v in vars(args).items()},
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_checkpoint": str(best_ckpt),
        "best_named_checkpoint": str(best_named_ckpt),
        "final_checkpoint": str(final_ckpt),
        "final_named_checkpoint": str(final_named_ckpt),
        "posterior_history": posterior_history,
        "before_uadal": before_metrics,
        "after_uadal": after_metrics,
        "openworld": openworld_metrics,
    }
    save_json_safe(summary, output_dir / "metrics.json")
    save_json_safe(summary, output_dir / "uadal_summary.json")

    flat_rows = []
    for group_name, group in [
        ("before_uadal", before_metrics),
        ("after_uadal", after_metrics),
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
        pd.DataFrame(flat_rows).to_csv(output_dir / "uadal_metrics_summary.csv", index=False)

    print("Saved UADAL outputs to", output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="UADAL open-set domain adaptation (Jang et al., NeurIPS 2022).")

    parser.add_argument("--standardized-csv", required=True)
    parser.add_argument("--source-dataset", default="PAD-UFES")
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--source-image-roots", nargs="+", required=True)
    parser.add_argument("--target-image-roots", nargs="+", required=True)
    parser.add_argument("--checkpoint", help="closed-set or DANN checkpoint to start from (recommended)")
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
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--sampler",
        choices=["notebook", "soft", "none"],
        default="notebook",
        help="Source-only class balancing (UADAL uses a class-balanced source loader).",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["known_macro_f1", "hos", "last"],
        default="known_macro_f1",
        help=(
            "known_macro_f1 matches train_dann.py (target_val known Macro-F1); "
            "hos uses target_val native HOS; last uses no target labels."
        ),
    )

    # UADAL-specific (defaults follow the official run_scripts.sh / config.py)
    parser.add_argument("--warmup-steps", type=int, default=500, help="source-only warm-up iterations (UADAL warmup_iter)")
    parser.add_argument("--bmm-update-every", type=int, default=2, help="epochs between posterior refits (UADAL update_term)")
    parser.add_argument("--ls-eps", type=float, default=0.1, help="label smoothing")
    parser.add_argument("--pseudo-threshold", type=float, default=0.85, help="confidence threshold for target pseudo-labels")
    parser.add_argument("--head-lr-mult", type=float, default=10.0, help="C/E/D lr = lr * this (UADAL uses 10x for heads)")
    parser.add_argument("--disc-clip", type=float, default=0.1, help="grad-norm clip for the discriminator")
    parser.add_argument("--disc-hidden", type=int, default=500, help="discriminator hidden width (UADAL bottle_neck_dim2)")
    parser.add_argument("--adv-weight", type=float, default=0.5)
    parser.add_argument("--pseudo-weight", type=float, default=0.5)
    parser.add_argument("--unknown-weight", type=float, default=0.2)

    parser.add_argument("--skip-openworld", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
