from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import ExperimentConfig, KNOWN_CLASSES, SEED
from .datasets import DomainAdaptationDataset, get_loss_weights_from_train_df, make_soft_weighted_random_sampler
from .metrics import evaluate_closed_set
from .models import build_dann_model
from .preprocessing import add_known_unknown_columns, load_standardized_splits
from .transforms import get_eval_transform, get_train_transform
from .utils import ensure_dir, get_device, load_model_state, save_checkpoint, save_json, seed_everything


def dann_lambda_schedule(step: int, total_steps: int, max_lambda: float = 1.0) -> float:
    p = step / max(total_steps, 1)
    return float(max_lambda * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))).item() - 1.0))


def train_dann_epoch(model, source_loader, target_loader, optimizer, class_criterion, domain_criterion, device, epoch, total_epochs, domain_loss_weight=1.0, max_lambda=1.0):
    model.train()
    total_steps = max(len(source_loader), len(target_loader)) * total_epochs
    source_iter = cycle(source_loader)
    target_iter = cycle(target_loader)
    n_steps = max(len(source_loader), len(target_loader))
    total = 0.0
    for local_step in tqdm(range(n_steps), desc="dann", leave=False):
        global_step = (epoch - 1) * n_steps + local_step
        lambd = dann_lambda_schedule(global_step, total_steps, max_lambda)
        src = next(source_iter)
        tgt = next(target_iter)
        src = {k: v.to(device) if hasattr(v, "to") else v for k, v in src.items()}
        tgt = {k: v.to(device) if hasattr(v, "to") else v for k, v in tgt.items()}
        optimizer.zero_grad(set_to_none=True)
        src_class_logits, src_domain_logits, _ = model(src["pixel_values"], src["input_ids"], src["attention_mask"], dann_lambda=lambd)
        _, tgt_domain_logits, _ = model(tgt["pixel_values"], tgt["input_ids"], tgt["attention_mask"], dann_lambda=lambd)
        class_loss = class_criterion(src_class_logits, src["label"])
        domain_logits = torch.cat([src_domain_logits, tgt_domain_logits], dim=0)
        domain_labels = torch.cat([src["domain"], tgt["domain"]], dim=0)
        domain_loss = domain_criterion(domain_logits, domain_labels)
        loss = class_loss + domain_loss_weight * domain_loss
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(n_steps, 1)


def run(args):
    seed_everything(SEED)
    cfg = ExperimentConfig()
    device = get_device()
    output_dir = ensure_dir(Path(args.output_dir) / args.model_family / args.text_col)
    image_model_name = cfg.resnet_model_name if args.model_family.startswith("resnet50") else cfg.image_model_name

    source_df = load_standardized_splits(args.standardized_csv, args.source_image_roots, args.source_dataset)
    target_df = load_standardized_splits(args.standardized_csv, args.target_image_roots, args.target_dataset)
    source_df = add_known_unknown_columns(source_df)
    target_df = add_known_unknown_columns(target_df)
    source_train_df = source_df[(source_df["split"].isin(["source_train", "train"])) & (~source_df["is_unknown"])].copy()
    target_adapt_df = target_df[(target_df["split"] == "target_adapt") & (~target_df["is_unknown"])].copy()
    target_val_df = target_df[(target_df["split"] == "target_val") & (~target_df["is_unknown"])].copy()

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    source_ds = DomainAdaptationDataset(source_train_df, tokenizer, get_train_transform(), args.text_col, domain_label=0, max_text_len=cfg.max_text_len)
    target_ds = DomainAdaptationDataset(target_adapt_df, tokenizer, get_eval_transform(), args.text_col, domain_label=1, max_text_len=cfg.max_text_len)
    val_ds = DomainAdaptationDataset(target_val_df, tokenizer, get_eval_transform(), args.text_col, domain_label=1, max_text_len=cfg.max_text_len)
    source_loader = DataLoader(source_ds, batch_size=args.batch_size, sampler=make_soft_weighted_random_sampler(source_train_df), num_workers=cfg.num_workers)
    target_loader = DataLoader(target_ds, batch_size=args.batch_size, sampler=make_soft_weighted_random_sampler(target_adapt_df), num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = build_dann_model(args.model_family, image_model_name, cfg.text_model_name, len(KNOWN_CLASSES), cfg.fusion_dim, cfg.num_heads).to(device)
    if args.checkpoint:
        load_model_state(model, args.checkpoint, strict=False)
    class_weights = get_loss_weights_from_train_df(source_train_df, len(KNOWN_CLASSES)).to(device)
    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_dann_epoch(model, source_loader, target_loader, optimizer, class_criterion, domain_criterion, device, epoch, args.epochs, cfg.domain_loss_weight, cfg.dann_lambda_max)
        val_metrics, _, _, _ = evaluate_closed_set(model, val_loader, class_criterion, device, len(KNOWN_CLASSES))
        row = {"epoch": epoch, "dann_loss": loss, **{f"target_val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(row)
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            save_checkpoint(model, output_dir / "best_dann.pt", {"epoch": epoch, "model_family": args.model_family, "text_col": args.text_col, "known_classes": KNOWN_CLASSES})
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    save_json({"best_target_val_f1_macro": best_val_f1}, output_dir / "metrics.json")
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
    parser.add_argument("--model-family", choices=["mobilevit_cross_attention", "mobilevit_gated", "resnet50_cross_attention", "resnet50_gated"], default="mobilevit_gated")
    parser.add_argument("--text-col", choices=["text_core", "text_full", "text_missing_explicit"], default="text_full")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
