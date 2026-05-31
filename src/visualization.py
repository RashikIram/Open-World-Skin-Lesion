from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from .transforms import IMAGENET_MEAN, IMAGENET_STD


# ============================================================
# General helpers
# ============================================================

def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _label_name(label_id: int, class_names: Sequence[str]) -> str:
    label_id = int(label_id)
    if 0 <= label_id < len(class_names):
        return str(class_names[label_id])
    return f"label_{label_id}"


def tensor_to_display_image(tensor):
    """
    Convert an ImageNet-normalized tensor [3, H, W] back to displayable RGB.
    """
    x = tensor.detach().cpu().clone()

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    x = x * std + mean
    x = x.clamp(0, 1)

    return x.permute(1, 2, 0).numpy()


# ============================================================
# Confusion matrix and ROC
# ============================================================

def plot_normalized_confusion_matrix(
    y_true,
    y_pred,
    class_names: Sequence[str],
    output_path: str | Path,
    title: str | None = None,
    values_format: str = ".4f",
    dpi: int = 300,
):
    """
    Plot a normalized confusion matrix using a fixed label list.

    The fixed labels are important. Without labels=list(range(n_classes)),
    sklearn can shrink the matrix when a class is absent from y_true/y_pred,
    which then breaks display_labels alignment.
    """
    output_path = _ensure_parent(output_path)

    labels = list(range(len(class_names)))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true",
    )
    cm = np.nan_to_num(cm, nan=0.0)

    fig, ax = plt.subplots(figsize=(8, 7))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(class_names),
    )
    disp.plot(
        ax=ax,
        values_format=values_format,
        xticks_rotation=45,
        colorbar=True,
    )

    ax.set_title(title or "Normalized confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return cm


def save_confusion_matrix_tables(
    y_true,
    y_pred,
    class_names: Sequence[str],
    output_dir: str | Path,
    prefix: str = "confusion_matrix",
):
    """
    Save count and normalized confusion matrices as CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(class_names)))

    cm_count = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )
    cm_norm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true",
    )
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)

    count_df = pd.DataFrame(
        cm_count,
        index=list(class_names),
        columns=list(class_names),
    )
    norm_df = pd.DataFrame(
        cm_norm,
        index=list(class_names),
        columns=list(class_names),
    )

    count_df.to_csv(output_dir / f"{prefix}_counts.csv")
    norm_df.to_csv(output_dir / f"{prefix}_normalized.csv")

    return count_df, norm_df


def plot_multiclass_roc(
    y_true,
    y_prob,
    class_names: Sequence[str],
    output_path: str | Path,
    title: str | None = None,
    dpi: int = 300,
):
    """
    Plot one-vs-rest ROC curves with fixed class ordering.

    Classes that have no positive or no negative examples are skipped because
    their ROC curve is undefined.
    """
    output_path = _ensure_parent(output_path)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    labels = list(range(len(class_names)))

    if y_prob.ndim != 2 or y_prob.shape[1] != len(class_names):
        raise ValueError(
            "y_prob must have shape [n_samples, n_classes] matching class_names. "
            f"Got y_prob.shape={y_prob.shape}, len(class_names)={len(class_names)}."
        )

    y_bin = label_binarize(y_true, classes=labels)
    if len(class_names) == 2 and y_bin.ndim == 2 and y_bin.shape[1] == 1:
        # sklearn returns one column for binary label_binarize.
        y_bin = np.column_stack([1 - y_bin[:, 0], y_bin[:, 0]])

    fig, ax = plt.subplots(figsize=(8, 7))
    rows = []

    for class_idx, class_name in enumerate(class_names):
        if class_idx >= y_bin.shape[1]:
            continue

        if len(np.unique(y_bin[:, class_idx])) < 2:
            continue

        fpr, tpr, thresholds = roc_curve(
            y_bin[:, class_idx],
            y_prob[:, class_idx],
        )
        class_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{class_name} AUC={class_auc:.4f}",
        )

        for fp, tp, thr in zip(fpr, tpr, thresholds):
            rows.append({
                "class": class_name,
                "fpr": float(fp),
                "tpr": float(tp),
                "threshold": float(thr),
                "auc": float(class_auc),
            })

    if rows:
        pd.DataFrame(rows).to_csv(
            output_path.with_name(output_path.stem + "_data.csv"),
            index=False,
        )

    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
    ax.set_title(title or "ROC curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(rows)


# ============================================================
# t-SNE
# ============================================================

@torch.no_grad()
def extract_fused_features(model, loader, device):
    """
    Extract fused features from a closed-set or DANN fusion model.

    Supports:
    - closed-set models with forward(..., return_features=True) -> logits, features
    - DANN models returning class_logits, domain_logits, fused_features
    """
    model.eval()

    all_features = []
    all_true = []
    all_pred = []
    all_prob = []

    for batch in tqdm(loader, desc="Extracting fused features", leave=False):
        batch = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in batch.items()
        }

        pixel_values = batch["pixel_values"]
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")

        try:
            out = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_features=True,
            )
        except TypeError:
            out = model(
                pixel_values,
                input_ids,
                attention_mask,
            )

        if isinstance(out, tuple):
            logits = out[0]
            if len(out) >= 3:
                features = out[-1]
            elif len(out) == 2:
                features = out[1]
            else:
                raise ValueError("Model returned an unsupported tuple output.")
        else:
            raise ValueError(
                "Feature extraction requires the model to return features. "
                "For closed-set models, forward(..., return_features=True) should be supported."
            )

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_features.append(features.detach().cpu().numpy())
        all_true.extend(batch["label"].detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())
        all_prob.append(probs.detach().cpu().numpy())

    return (
        np.concatenate(all_features, axis=0),
        np.asarray(all_true),
        np.asarray(all_pred),
        np.concatenate(all_prob, axis=0),
    )


def plot_tsne(
    features,
    labels,
    class_names: Sequence[str],
    output_path: str | Path,
    perplexity: int = 30,
    title: str | None = None,
    coordinate_output_path: str | Path | None = None,
    pred_labels=None,
    random_state: int = 42,
    dpi: int = 300,
):
    """
    Plot t-SNE and optionally save coordinates.

    This keeps the original function name but adds coordinate saving so the
    notebook-style t-SNE CSV artifact can be produced.
    """
    output_path = _ensure_parent(output_path)

    features = np.asarray(features)
    labels = np.asarray(labels)

    if features.ndim != 2:
        raise ValueError(f"features must be 2D. Got shape={features.shape}.")

    n_samples = features.shape[0]
    if n_samples < 3:
        print("Skipping t-SNE because there are fewer than 3 samples.")
        return pd.DataFrame()

    perplexity = min(perplexity, max(2, (n_samples - 1) // 3))

    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(features)

    tsne_df = pd.DataFrame({
        "tsne_1": emb[:, 0],
        "tsne_2": emb[:, 1],
        "true_label_id": labels,
        "true_label": [_label_name(x, class_names) for x in labels],
    })

    if pred_labels is not None:
        pred_labels = np.asarray(pred_labels)
        tsne_df["pred_label_id"] = pred_labels
        tsne_df["pred_label"] = [_label_name(x, class_names) for x in pred_labels]

    if coordinate_output_path is None:
        coordinate_output_path = output_path.with_name(output_path.stem + "_coordinates.csv")

    coordinate_output_path = _ensure_parent(coordinate_output_path)
    tsne_df.to_csv(coordinate_output_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 7))

    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if np.any(mask):
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=12,
                alpha=0.75,
                label=class_name,
            )

    ax.set_title(title or "t-SNE of fused features")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return tsne_df


def plot_tsne_from_loader(
    model,
    loader,
    device,
    class_names: Sequence[str],
    output_dir: str | Path,
    prefix: str = "test",
    perplexity: int = 30,
):
    """
    Extract fused features from a model/loader and save notebook-style t-SNE
    coordinate CSV plus PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, y_true, y_pred, y_prob = extract_fused_features(
        model,
        loader,
        device,
    )

    np.save(output_dir / f"{prefix}_fused_features.npy", features)
    pd.DataFrame(y_prob, columns=[f"prob_{c}" for c in class_names]).to_csv(
        output_dir / f"{prefix}_feature_probs.csv",
        index=False,
    )

    tsne_df = plot_tsne(
        features=features,
        labels=y_true,
        pred_labels=y_pred,
        class_names=class_names,
        output_path=output_dir / f"{prefix}_tsne.png",
        coordinate_output_path=output_dir / f"{prefix}_tsne_coordinates.csv",
        perplexity=perplexity,
        title=f"{prefix.replace('_', ' ').title()} t-SNE",
    )

    return tsne_df


# ============================================================
# GradCAM++
# ============================================================

def compute_gradcampp_from_feature_map(
    activations,
    gradients=None,
    target_score=None,
    eps: float = 1e-8,
):
    """
    Compute GradCAM++ from a final convolutional feature map.

    Preferred notebook-style usage:
        cam = compute_gradcampp_from_feature_map(activations, gradients)

    Backwards-compatible usage:
        cam = compute_gradcampp_from_feature_map(feature_map, target_score=score)

    Expected shapes:
    - activations [C, H, W] or [1, C, H, W]
    - gradients   [C, H, W] or [1, C, H, W]

    Returns:
    - cam tensor [H, W], normalized to [0, 1]
    """
    if gradients is None:
        if target_score is None:
            raise ValueError("Provide either gradients or target_score.")
        gradients = torch.autograd.grad(
            target_score,
            activations,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

    if activations.ndim == 4:
        activations = activations[0]
    if gradients.ndim == 4:
        gradients = gradients[0]

    activations = activations.detach()
    gradients = gradients.detach()

    # Notebook GradCAM++ formula.
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


def _model_logits(model, pixel_values, input_ids=None, attention_mask=None):
    out = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    if isinstance(out, tuple):
        return out[0]
    return out


def generate_gradcampp_examples(
    model,
    dataset,
    output_dir: str | Path,
    class_names: Sequence[str],
    device=None,
    n_examples: int = 6,
    select_mode: str = "first_per_class",
    prefix: str = "gradcampp",
    target: str = "predicted",
    alpha: float = 0.45,
    dpi: int = 300,
):
    """
    Generate notebook-style GradCAM++ overlays.

    Requirements:
    - model.image_encoder.last_feature_map must be set during forward.
    - the image encoder must call retain_grad() on the final feature map.

    Parameters:
    - select_mode="first_per_class": first available sample from each class.
    - target="predicted": backprop through predicted class score.
    - target="true": backprop through true class score.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    if not hasattr(model, "image_encoder"):
        raise AttributeError("GradCAM++ requires model.image_encoder.last_feature_map.")

    labels = None
    if hasattr(dataset, "df") and "label_id" in dataset.df.columns:
        labels = dataset.df["label_id"].to_numpy()
    elif hasattr(dataset, "df") and "label" in dataset.df.columns:
        labels = dataset.df["label"].to_numpy()

    if select_mode == "first_per_class" and labels is not None:
        selected_indices = []
        for class_idx in range(len(class_names)):
            idxs = np.where(labels == class_idx)[0]
            if len(idxs) > 0:
                selected_indices.append(int(idxs[0]))
        selected_indices = selected_indices[:n_examples]
    else:
        selected_indices = list(range(min(n_examples, len(dataset))))

    saved_paths = []

    for saved_idx, dataset_idx in enumerate(selected_indices):
        sample = dataset[dataset_idx]

        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
        pixel_values.requires_grad_(True)

        input_ids = sample.get("input_ids")
        if input_ids is not None:
            input_ids = input_ids.unsqueeze(0).to(device)

        attention_mask = sample.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0).to(device)

        true_label = int(sample["label"].item())

        model.zero_grad(set_to_none=True)

        logits = _model_logits(
            model,
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probs = torch.softmax(logits, dim=1)
        pred_label = int(torch.argmax(probs, dim=1).item())

        if target == "true":
            target_label = true_label
        elif target == "predicted":
            target_label = pred_label
        else:
            raise ValueError("target must be 'predicted' or 'true'.")

        target_score = logits[0, target_label]
        target_score.backward(retain_graph=True)

        feature_map = model.image_encoder.last_feature_map

        if feature_map is None or feature_map.grad is None:
            print(
                "Skipping GradCAM++ example because feature-map gradients were unavailable. "
                "Check that the image encoder stores last_feature_map and calls retain_grad()."
            )
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

        true_name = _label_name(true_label, class_names)
        pred_name = _label_name(pred_label, class_names)

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(image_np)
        plt.imshow(cam_np, cmap="jet", alpha=alpha)
        plt.axis("off")
        plt.title(
            f"True: {true_name} | Pred: {pred_name} | "
            f"P={probs[0, pred_label].item():.4f}"
        )

        out_path = output_dir / (
            f"{prefix}_{saved_idx:02d}_"
            f"idx_{dataset_idx}_"
            f"true_{true_name}_"
            f"pred_{pred_name}.png"
        )

        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(str(out_path))

    pd.DataFrame({
        "path": saved_paths,
    }).to_csv(output_dir / f"{prefix}_saved_files.csv", index=False)

    print(f"Saved {len(saved_paths)} GradCAM++ examples to: {output_dir}")

    return saved_paths
