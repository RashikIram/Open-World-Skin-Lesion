from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from .transforms import IMAGENET_MEAN, IMAGENET_STD


def plot_normalized_confusion_matrix(y_true, y_pred, class_names, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, values_format=".2f", xticks_rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_multiclass_roc(y_true, y_prob, class_names, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    y_bin = label_binarize(y_true, classes=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{name} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_tsne(features, labels, class_names, output_path, perplexity: int = 30):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = np.asarray(features)
    labels = np.asarray(labels)
    if len(features) < 3:
        return
    perplexity = min(perplexity, max(2, len(features) // 3))
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(features)
    fig, ax = plt.subplots(figsize=(8, 7))
    for idx, name in enumerate(class_names):
        mask = labels == idx
        if mask.any():
            ax.scatter(emb[mask, 0], emb[mask, 1], s=10, label=name, alpha=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def tensor_to_display_image(tensor):
    x = tensor.detach().cpu().clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = x * std + mean
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    return x


def compute_gradcampp_from_feature_map(feature_map, target_score):
    grads = torch.autograd.grad(target_score, feature_map, retain_graph=True, create_graph=False)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feature_map).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam
