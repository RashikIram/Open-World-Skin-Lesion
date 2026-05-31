from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import ExperimentConfig, KNOWN_CLASSES, SEED
from .datasets import (
    PadUfesClosedSetDataset,
    get_loss_weights_from_train_df,
    make_soft_weighted_random_sampler,
)
from .metrics import evaluate_closed_set
from .models import build_model
from .preprocessing import prepare_padufes_dataframe, stratified_train_val_test_split
from .transforms import IMAGENET_MEAN, IMAGENET_STD, get_eval_transform, get_train_transform
from .utils import ensure_dir, get_device, save_checkpoint, save_json, seed_everything
from .visualization import plot_multiclass_roc, plot_normalized_confusion_matrix


TEXT_EXPERIMENTS = [
    "text_full",
    "text_core",
    "text_missing_explicit",
]


MODEL_SAVE_STEMS = {
    "mobilevit_cross_attention": "cross_attention_mobile_vit",
    "mobilevit_gated": "gated_mobile_vit",
    "mobilevit_concat": "concat_mobile_vit",
    "resnet50_cross_attention": "cross_attention_resnet50",
    "resnet50_gated": "gated_resnet50",
    "resnet50_concat": "concat_resnet50",
    "image_only": "image_only",
}


def model_save_stem(model_family: str) -> str:
    return MODEL_SAVE_STEMS.get(model_family, model_family)


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in batch.items()
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            batch["pixel_values"],
            batch.get("input_ids"),
            batch.get("attention_mask"),
        )

        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()

        total += float(loss.item())

    return total / max(len(loader), 1)


def tensor_to_display_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert an ImageNet-normalized CHW tensor back to a displayable HWC image.
    """
    x = tensor.detach().cpu().clone()

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    x = (x * std + mean).clamp(0, 1)

    return x.permute(1, 2, 0).numpy()


def compute_gradcampp_from_feature_map(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    GradCAM++ calculation matching the closed-set notebook implementation.

    Parameters
    ----------
    activations:
        Feature map for one sample, shape [C, H, W].
    gradients:
        Gradients of the selected class score w.r.t. activations, shape [C, H, W].
    """
    grad_2 = gradients.pow(2)
    grad_3 = gradients.pow(3)

    spatial_sum = torch.sum(
        activations * grad_3,
        dim=(1, 2),
        keepdim=True,
    )

    alpha = grad_2 / (2.0 * grad_2 + spatial_sum + eps)
    alpha = torch.where(
        torch.isfinite(alpha),
        alpha,
        torch.zeros_like(alpha),
    )

    weights = torch.sum(
        alpha * F.relu(gradients),
        dim=(1, 2),
    )

    cam = torch.sum(
        weights[:, None, None] * activations,
        dim=0,
    )

    cam = F.relu(cam)

    cam_min = cam.min()
    cam_max = cam.max()

    if (cam_max - cam_min) > eps:
        cam = (cam - cam_min) / (cam_max - cam_min + eps)
    else:
        cam = torch.zeros_like(cam)

    return cam


def generate_gradcampp_examples(
    model,
    dataset,
    output_dir: str | Path,
    device: torch.device,
    n_examples: int = 6,
) -> None:
    """
    Save one GradCAM++ visualisation per class, up to n_examples total.
    """
    output_dir = ensure_dir(output_dir)

    if not hasattr(model, "image_encoder"):
        print("Skipping GradCAM++ because the model has no image_encoder attribute.")
        return

    model.eval()

    selected_indices: list[int] = []
    labels = dataset.df["label_id"].to_numpy()

    for class_idx in range(len(KNOWN_CLASSES)):
        idxs = np.where(labels == class_idx)[0]
        if len(idxs) > 0:
            selected_indices.append(int(idxs[0]))

    selected_indices = selected_indices[:n_examples]

    saved = 0

    for dataset_idx in selected_indices:
        sample = dataset[dataset_idx]

        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
        pixel_values.requires_grad_(True)

        input_ids = sample.get("input_ids")
        attention_mask = sample.get("attention_mask")

        if input_ids is not None:
            input_ids = input_ids.unsqueeze(0).to(device)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0).to(device)

        true_label = int(sample["label"].item())

        model.zero_grad(set_to_none=True)

        logits = model(
            pixel_values,
            input_ids,
            attention_mask,
        )

        probs = torch.softmax(logits, dim=1)
        pred_label = int(torch.argmax(probs, dim=1).item())

        target_score = logits[0, pred_label]
        target_score.backward(retain_graph=True)

        feature_map = getattr(model.image_encoder, "last_feature_map", None)

        if feature_map is None or feature_map.grad is None:
            print("Skipping GradCAM++ example because gradients were unavailable.")
            continue

        activations = feature_map.detach()[0]
        gradients = feature_map.grad.detach()[0]

        cam = compute_gradcampp_from_feature_map(
            activations=activations,
            gradients=gradients,
        )

        cam = F.interpolate(
            cam[None, None, :, :],
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam_np = cam.detach().cpu().numpy()
        image_np = tensor_to_display_image(sample["pixel_values"])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_np)
        ax.imshow(cam_np, cmap="jet", alpha=0.45)
        ax.axis("off")
        ax.set_title(
            f"True: {KNOWN_CLASSES[true_label]} | "
            f"Pred: {KNOWN_CLASSES[pred_label]} | "
            f"P={probs[0, pred_label].item():.4f}"
        )

        out_path = output_dir / (
            f"gradcampp_{saved:02d}_"
            f"true_{KNOWN_CLASSES[true_label]}_"
            f"pred_{KNOWN_CLASSES[pred_label]}.png"
        )

        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        saved += 1

    print(f"Saved {saved} GradCAM++ examples to: {output_dir}")


@torch.no_grad()
def extract_fused_features(model, loader, device: torch.device):
    model.eval()

    all_features = []
    all_true = []
    all_pred = []

    for batch in tqdm(loader, desc="extract features", leave=False):
        batch = batch_to_device(batch, device)

        output = model(
            batch["pixel_values"],
            batch.get("input_ids"),
            batch.get("attention_mask"),
            return_features=True,
        )

        logits, features = output
        preds = logits.argmax(dim=1)

        all_features.append(features.detach().cpu().numpy())
        all_true.extend(batch["label"].detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())

    return (
        np.concatenate(all_features, axis=0),
        np.asarray(all_true),
        np.asarray(all_pred),
    )


def save_tsne_outputs(
    model,
    loader,
    output_dir: str | Path,
    device: torch.device,
    title: str,
) -> None:
    """
    Save both t-SNE coordinates and a t-SNE plot for the test split.
    """
    output_dir = ensure_dir(output_dir)

    features, y_true, y_pred = extract_fused_features(model, loader, device)

    n_samples = features.shape[0]
    if n_samples < 3:
        print("Skipping t-SNE because there are fewer than 3 samples.")
        return

    perplexity = min(30, max(2, (n_samples - 1) // 3))

    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=SEED,
    ).fit_transform(features)

    tsne_df = pd.DataFrame({
        "tsne_1": emb[:, 0],
        "tsne_2": emb[:, 1],
        "true_label_id": y_true,
        "true_label": [KNOWN_CLASSES[int(x)] for x in y_true],
        "pred_label_id": y_pred,
        "pred_label": [KNOWN_CLASSES[int(x)] for x in y_pred],
    })

    tsne_df.to_csv(output_dir / "test_tsne_coordinates.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 7))

    for class_idx, class_name in enumerate(KNOWN_CLASSES):
        mask = y_true == class_idx
        if mask.any():
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=16,
                alpha=0.75,
                label=class_name,
            )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "test_tsne.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_classification_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    test_df: pd.DataFrame,
    output_dir: str | Path,
    metrics: dict,
) -> None:
    output_dir = ensure_dir(output_dir)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(KNOWN_CLASSES))),
        target_names=KNOWN_CLASSES,
        zero_division=0,
        digits=4,
        output_dict=True,
    )

    pd.DataFrame(report_dict).transpose().round(4).to_csv(
        output_dir / "test_classification_report.csv"
    )

    pred_df = test_df.reset_index(drop=True).copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["true_label"] = [KNOWN_CLASSES[int(x)] for x in y_true]
    pred_df["pred_label"] = [KNOWN_CLASSES[int(x)] for x in y_pred]

    for class_idx, class_name in enumerate(KNOWN_CLASSES):
        pred_df[f"prob_{class_name}"] = y_prob[:, class_idx]

    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    save_json(metrics, output_dir / "test_metrics.json")
    save_json(metrics, output_dir / "metrics.json")


def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    text_col: str,
    batch_size: int,
    cfg: ExperimentConfig,
):
    train_ds = PadUfesClosedSetDataset(
        train_df,
        tokenizer,
        get_train_transform(),
        text_col,
        max_text_len=cfg.max_text_len,
    )

    val_ds = PadUfesClosedSetDataset(
        val_df,
        tokenizer,
        get_eval_transform(),
        text_col,
        max_text_len=cfg.max_text_len,
    )

    test_ds = PadUfesClosedSetDataset(
        test_df,
        tokenizer,
        get_eval_transform(),
        text_col,
        max_text_len=cfg.max_text_len,
    )

    sampler = (
        make_soft_weighted_random_sampler(train_df, beta=cfg.balance_beta)
        if cfg.use_soft_weighted_sampler
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def run_one_text_experiment(
    args,
    cfg: ExperimentConfig,
    device: torch.device,
    tokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
) -> dict:
    output_base = ensure_dir(args.output_dir)
    output_dir = ensure_dir(output_base / args.model_family / text_col)

    stem = model_save_stem(args.model_family)

    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    _, _, test_ds, train_loader, val_loader, test_loader = make_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        text_col=text_col,
        batch_size=args.batch_size,
        cfg=cfg,
    )

    image_model_name = (
        cfg.resnet_model_name
        if args.model_family.startswith("resnet50")
        else cfg.image_model_name
    )

    model = build_model(
        args.model_family,
        image_model_name,
        cfg.text_model_name,
        len(KNOWN_CLASSES),
        cfg.fusion_dim,
        cfg.num_heads,
        freeze_backbones=getattr(cfg, "freeze_backbones", False),
    ).to(device)

    loss_weights = (
        get_loss_weights_from_train_df(
            train_df,
            len(KNOWN_CLASSES),
            cfg.balance_beta,
        ).to(device)
        if cfg.use_weighted_loss
        else None
    )

    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=getattr(cfg, "scheduler_factor", 0.5),
        patience=getattr(cfg, "scheduler_patience", 3),
    )

    best_val_f1 = -1.0
    patience_left = cfg.patience
    history: list[dict] = []

    best_payload_extra = {
        "model_family": args.model_family,
        "text_col": text_col,
        "known_classes": KNOWN_CLASSES,
        "image_model_name": image_model_name,
        "text_model_name": cfg.text_model_name,
        "fusion_dim": cfg.fusion_dim,
        "num_heads": cfg.num_heads,
    }

    best_named_path = output_dir / f"{stem}_{text_col}_best.pt"
    best_short_path = output_dir / "best.pt"
    root_best_path = output_base / f"{stem}_{text_col}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_metrics, _, _, _ = evaluate_closed_set(
            model,
            val_loader,
            criterion,
            device,
            len(KNOWN_CLASSES),
        )

        val_f1 = float(
            val_metrics.get(
                "macro_f1",
                val_metrics.get("f1_macro", float("nan")),
            )
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        history.append(row)
        print(row)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_left = cfg.patience

            payload = {
                **best_payload_extra,
                "epoch": epoch,
                "best_val_f1_macro": best_val_f1,
            }

            save_checkpoint(model, best_short_path, payload)
            save_checkpoint(model, best_named_path, payload)
            save_checkpoint(model, root_best_path, payload)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)

    checkpoint = torch.load(best_short_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, y_true, y_pred, y_prob = evaluate_closed_set(
        model,
        test_loader,
        criterion,
        device,
        len(KNOWN_CLASSES),
    )

    save_classification_outputs(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        test_df=test_df,
        output_dir=output_dir,
        metrics=test_metrics,
    )

    plot_normalized_confusion_matrix(
        y_true,
        y_pred,
        KNOWN_CLASSES,
        output_dir / "test_confusion_matrix.png",
    )

    # Backwards-compatible filename used by the earlier modular script.
    plot_normalized_confusion_matrix(
        y_true,
        y_pred,
        KNOWN_CLASSES,
        output_dir / "confusion_matrix.png",
    )

    plot_multiclass_roc(
        y_true,
        y_prob,
        KNOWN_CLASSES,
        output_dir / "test_roc_curves.png",
    )

    # Backwards-compatible filename used by the earlier modular script.
    plot_multiclass_roc(
        y_true,
        y_prob,
        KNOWN_CLASSES,
        output_dir / "roc_curves.png",
    )

    save_tsne_outputs(
        model,
        test_loader,
        output_dir,
        device,
        title=f"Closed-set t-SNE: {args.model_family} / {text_col}",
    )

    if not args.skip_gradcam:
        generate_gradcampp_examples(
            model,
            test_ds,
            output_dir / "gradcampp_examples",
            device,
            n_examples=args.gradcam_examples,
        )

    final_payload = {
        **best_payload_extra,
        "best_checkpoint": str(best_named_path),
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "test_metrics": test_metrics,
    }

    final_named_path = output_dir / f"{stem}_{text_col}_final.pt"
    final_short_path = output_dir / "final.pt"
    root_final_path = output_base / f"{stem}_{text_col}_final.pt"

    save_checkpoint(model, final_short_path, final_payload)
    save_checkpoint(model, final_named_path, final_payload)
    save_checkpoint(model, root_final_path, final_payload)

    summary_row = {
        "model_family": args.model_family,
        "text_col": text_col,
        "output_dir": str(output_dir),
        "best_checkpoint": str(best_named_path),
        "root_best_checkpoint": str(root_best_path),
        "final_checkpoint": str(final_named_path),
        "best_val_f1_macro": best_val_f1,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

    print("Saved closed-set outputs to", output_dir)

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary_row


def resolve_text_experiments(args) -> list[str]:
    if args.model_family == "image_only":
        if args.text_col == "all":
            return ["image_only"]
        return [args.text_col]

    if args.text_col == "all":
        return TEXT_EXPERIMENTS

    return [args.text_col]

def cleanup_cuda(*objects):
    import gc
    for obj in objects:
        try:
            del obj
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()

def run(args):
    seed_everything(SEED)

    device = get_device()
    cfg = ExperimentConfig()
    output_base = ensure_dir(args.output_dir)

    df = prepare_padufes_dataframe(
        args.padufes_csv,
        args.padufes_image_dir,
    )

    train_df, val_df, test_df = stratified_train_val_test_split(df)

    # Save the shared split once as well as inside each experiment folder.
    train_df.to_csv(output_base / "train_split.csv", index=False)
    val_df.to_csv(output_base / "val_split.csv", index=False)
    test_df.to_csv(output_base / "test_split.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    text_experiments = resolve_text_experiments(args)

    summary_rows = []

    for text_col in text_experiments:
        print("\n" + "=" * 80)
        print(f"Closed-set experiment: model_family={args.model_family}, text_col={text_col}")
        print("=" * 80)

        try:
            summary = run_one_text_experiment(
                args=args,
                cfg=cfg,
                device=device,
                tokenizer=tokenizer,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                text_col=text_col,
            )
            summary_rows.append(summary)

        finally:
            cleanup_cuda()

    summary_df = pd.DataFrame(summary_rows)

    stem = model_save_stem(args.model_family)
    summary_df.to_csv(output_base / f"{stem}_closed_set_summary.csv", index=False)
    summary_df.to_csv(output_base / "closed_set_summary.csv", index=False)

    print("\nClosed-set summary:")
    print(summary_df)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--padufes-csv", required=True)
    parser.add_argument("--padufes-image-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--model-family",
        choices=[
            "image_only",
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
        choices=[
            "all",
            "image_only",
            "text_core",
            "text_full",
            "text_missing_explicit",
        ],
        default="all",
        help=(
            "Use 'all' to run the three markdown text experiments: "
            "text_full, text_core, text_missing_explicit."
        ),
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument(
        "--gradcam-examples",
        type=int,
        default=6,
        help="Maximum number of GradCAM++ examples to save.",
    )

    parser.add_argument(
        "--skip-gradcam",
        action="store_true",
        help="Skip GradCAM++ image generation.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
