from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import ExperimentConfig, KNOWN_CLASSES, UNKNOWN_LABEL_NAME, SEED
from .datasets import OpenWorldDataset
from .metrics import compute_open_world_metrics, energy_score, find_best_unknown_threshold, predict_open_world_from_threshold
from .models import build_dann_model, build_model
from .preprocessing import add_known_unknown_columns, load_standardized_splits
from .transforms import get_eval_transform
from .utils import ensure_dir, get_device, load_model_state, save_json, seed_everything
from .visualization import plot_normalized_confusion_matrix


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
    return np.concatenate(logits_all, axis=0), np.asarray(labels_all), np.asarray(indices_all)


def evaluate_split(model, df, tokenizer, text_col, batch_size, device, threshold=None):
    ds = OpenWorldDataset(df, tokenizer, get_eval_transform(), text_col)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    logits, y_open, indices = collect_outputs(model, loader, device)
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    known_pred = probs.argmax(axis=1)
    # Higher score should mean more likely unknown. Energy is often lower for confident known samples,
    # so use negative energy as a monotonic unknownness score.
    unknown_score = energy_score(logits)
    unknown_id = len(KNOWN_CLASSES)
    if threshold is None:
        best = find_best_unknown_threshold(y_open, known_pred, unknown_score, unknown_id)
        threshold = best["threshold"]
    pred_open = predict_open_world_from_threshold(known_pred, unknown_score, threshold, unknown_id)
    metrics = compute_open_world_metrics(y_open, pred_open, unknown_score, unknown_id)
    metrics["threshold"] = float(threshold)
    pred_df = df.iloc[indices].copy().reset_index(drop=True)
    pred_df["y_open_true"] = y_open
    pred_df["known_pred"] = known_pred
    pred_df["unknown_score"] = unknown_score
    pred_df["y_open_pred"] = pred_open
    return metrics, pred_df


def run(args):
    seed_everything(SEED)
    cfg = ExperimentConfig()
    device = get_device()
    output_dir = ensure_dir(Path(args.output_dir) / args.model_family / args.text_col)
    image_model_name = cfg.resnet_model_name if args.model_family.startswith("resnet50") else cfg.image_model_name

    df = load_standardized_splits(args.standardized_csv, args.target_image_roots, args.target_dataset)
    df = add_known_unknown_columns(df)
    val_df = df[df["split"] == "target_val"].reset_index(drop=True)
    test_df = df[df["split"] == "target_test"].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    builder = build_dann_model if args.dann else build_model
    model = builder(args.model_family, image_model_name, cfg.text_model_name, len(KNOWN_CLASSES), cfg.fusion_dim, cfg.num_heads).to(device)
    load_model_state(model, args.checkpoint, strict=False)

    val_metrics, val_pred = evaluate_split(model, val_df, tokenizer, args.text_col, args.batch_size, device, threshold=None)
    test_metrics, test_pred = evaluate_split(model, test_df, tokenizer, args.text_col, args.batch_size, device, threshold=val_metrics["threshold"])

    save_json({"val": val_metrics, "test": test_metrics}, output_dir / "openworld_metrics.json")
    val_pred.to_csv(output_dir / "val_openworld_predictions.csv", index=False)
    test_pred.to_csv(output_dir / "test_openworld_predictions.csv", index=False)
    class_names = KNOWN_CLASSES + [UNKNOWN_LABEL_NAME]
    plot_normalized_confusion_matrix(test_pred["y_open_true"], test_pred["y_open_pred"], class_names, output_dir / "openworld_confusion_matrix.png")
    print("Saved open-world outputs to", output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardized-csv", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--target-image-roots", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-family", choices=["mobilevit_cross_attention", "mobilevit_gated", "resnet50_cross_attention", "resnet50_gated"], default="mobilevit_gated")
    parser.add_argument("--text-col", choices=["text_core", "text_full", "text_missing_explicit"], default="text_full")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dann", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
