from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import ExperimentConfig, KNOWN_CLASSES, SEED
from .datasets import PadUfesClosedSetDataset, get_loss_weights_from_train_df, make_soft_weighted_random_sampler
from .metrics import evaluate_closed_set
from .models import build_model
from .preprocessing import prepare_padufes_dataframe, stratified_train_val_test_split
from .transforms import get_eval_transform, get_train_transform
from .utils import ensure_dir, get_device, save_checkpoint, save_json, seed_everything
from .visualization import plot_multiclass_roc, plot_normalized_confusion_matrix


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["pixel_values"], batch.get("input_ids"), batch.get("attention_mask"))
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


def run(args):
    seed_everything(SEED)
    device = get_device()
    cfg = ExperimentConfig()
    output_dir = ensure_dir(Path(args.output_dir) / args.model_family / args.text_col)

    image_model_name = cfg.resnet_model_name if args.model_family.startswith("resnet50") else cfg.image_model_name
    df = prepare_padufes_dataframe(args.padufes_csv, args.padufes_image_dir)
    train_df, val_df, test_df = stratified_train_val_test_split(df)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    train_ds = PadUfesClosedSetDataset(train_df, tokenizer, get_train_transform(), args.text_col, max_text_len=cfg.max_text_len)
    val_ds = PadUfesClosedSetDataset(val_df, tokenizer, get_eval_transform(), args.text_col, max_text_len=cfg.max_text_len)
    test_ds = PadUfesClosedSetDataset(test_df, tokenizer, get_eval_transform(), args.text_col, max_text_len=cfg.max_text_len)

    sampler = make_soft_weighted_random_sampler(train_df, beta=cfg.balance_beta) if cfg.use_soft_weighted_sampler else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = build_model(args.model_family, image_model_name, cfg.text_model_name, len(KNOWN_CLASSES), cfg.fusion_dim, cfg.num_heads).to(device)
    loss_weights = get_loss_weights_from_train_df(train_df, len(KNOWN_CLASSES), cfg.balance_beta).to(device) if cfg.use_weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_f1 = -1.0
    patience_left = cfg.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _, _ = evaluate_closed_set(model, val_loader, criterion, device, len(KNOWN_CLASSES))
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(row)
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            patience_left = cfg.patience
            save_checkpoint(model, output_dir / "best.pt", {"epoch": epoch, "model_family": args.model_family, "text_col": args.text_col, "known_classes": KNOWN_CLASSES})
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, y_true, y_pred, y_prob = evaluate_closed_set(model, test_loader, criterion, device, len(KNOWN_CLASSES))
    save_json(test_metrics, output_dir / "metrics.json")
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(output_dir / "test_predictions.csv", index=False)
    plot_normalized_confusion_matrix(y_true, y_pred, KNOWN_CLASSES, output_dir / "confusion_matrix.png")
    plot_multiclass_roc(y_true, y_prob, KNOWN_CLASSES, output_dir / "roc_curves.png")
    print("Saved closed-set outputs to", output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--padufes-csv", required=True)
    parser.add_argument("--padufes-image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-family", choices=["image_only", "mobilevit_cross_attention", "mobilevit_gated", "resnet50_cross_attention", "resnet50_gated"], default="mobilevit_gated")
    parser.add_argument("--text-col", choices=["image_only", "text_core", "text_full", "text_missing_explicit"], default="text_full")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
