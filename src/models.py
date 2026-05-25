from __future__ import annotations

from typing import Literal

import timm
import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import AutoModel


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return GradReverse.apply(x, lambd)


class MobileViTAdapter(nn.Module):
    """
    MobileViT image encoder used by the MobileViT fusion models.

    This matches the original notebook implementation:
    - loads a timm MobileViT backbone with features_only=True
    - keeps the final feature map for Grad-CAM++
    - converts the final feature map into image tokens
    - prepends a global image token
    """

    def __init__(self, model_name: str = "mobilevit_s.cvnets_in1k"):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
        )

        self.last_feature_map = None

    def forward(self, x):
        feats = self.backbone(x)
        last = feats[-1]

        # Needed for Grad-CAM++.
        self.last_feature_map = last
        if last.requires_grad:
            last.retain_grad()

        b, c, h, w = last.shape

        spatial = (
            last.reshape(b, c, h * w)
            .permute(0, 2, 1)
            .contiguous()
        )

        global_token = (
            last.mean(dim=(2, 3))
            .unsqueeze(1)
        )

        return torch.cat(
            [global_token, spatial],
            dim=1,
        )


class ResNet50Adapter(nn.Module):
    """
    ResNet-50 image encoder used by the ResNet fusion models.

    It uses the same tokenisation style as MobileViT:
    final feature map -> spatial tokens + global token.
    """

    def __init__(self, model_name: str = "resnet50"):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
        )

        self.last_feature_map = None

    def forward(self, x):
        feats = self.backbone(x)
        last = feats[-1]

        self.last_feature_map = last
        if last.requires_grad:
            last.retain_grad()

        b, c, h, w = last.shape

        spatial = (
            last.reshape(b, c, h * w)
            .permute(0, 2, 1)
            .contiguous()
        )

        global_token = (
            last.mean(dim=(2, 3))
            .unsqueeze(1)
        )

        return torch.cat(
            [global_token, spatial],
            dim=1,
        )


class FeatureMapAdapter(nn.Module):
    """
    Generic image-only adapter.

    Kept for ImageOnlyClassifier and any future timm backbone.
    """

    def __init__(self, model_name: str):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
        )

        self.last_feature_map = None

    def forward(self, x):
        feats = self.backbone(x)
        last = feats[-1]

        self.last_feature_map = last
        if last.requires_grad:
            last.retain_grad()

        b, c, h, w = last.shape

        spatial = (
            last.reshape(b, c, h * w)
            .permute(0, 2, 1)
            .contiguous()
        )

        global_token = (
            last.mean(dim=(2, 3))
            .unsqueeze(1)
        )

        return torch.cat(
            [global_token, spatial],
            dim=1,
        )


class CrossAttentionFusionHead(nn.Module):
    def __init__(self, image_hidden: int, text_hidden: int, fusion_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.image_proj = nn.Linear(image_hidden, fusion_dim)
        self.text_proj = nn.Linear(text_hidden, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, image_tokens, text_tokens, attention_mask):
        image_tokens = self.image_proj(image_tokens)
        text_tokens = self.text_proj(text_tokens)
        fused_tokens, _ = self.cross_attn(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=(attention_mask == 0),
        )
        fused_tokens = self.norm(fused_tokens + image_tokens)
        return fused_tokens[:, 0, :]


class GatedFusionHead(nn.Module):
    def __init__(self, image_hidden: int, text_hidden: int, fusion_dim: int = 256):
        super().__init__()
        self.image_proj = nn.Linear(image_hidden, fusion_dim)
        self.text_proj = nn.Linear(text_hidden, fusion_dim)
        self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, image_tokens, text_tokens, attention_mask=None):
        image_feat = self.image_proj(image_tokens[:, 0, :])
        text_feat = self.text_proj(text_tokens[:, 0, :])
        gate = self.gate(torch.cat([image_feat, text_feat], dim=1))
        fused_cls = gate * image_feat + (1.0 - gate) * text_feat
        return self.norm(fused_cls)


def make_classifier(fusion_dim: int, num_classes: int, dropout: float = 0.3):
    return nn.Sequential(
        nn.Linear(fusion_dim, fusion_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fusion_dim, num_classes),
    )


class TextImageFusionClosedSet(nn.Module):
    def __init__(self, image_encoder: nn.Module, text_model_name: str, num_classes: int, fusion: Literal["cross_attention", "gated"], fusion_dim: int = 256, num_heads: int = 4, freeze_backbones: bool = False):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            image_hidden = self.image_encoder(dummy).shape[-1]
        text_hidden = self.text_encoder.config.hidden_size
        if fusion == "cross_attention":
            self.fusion = CrossAttentionFusionHead(image_hidden, text_hidden, fusion_dim, num_heads)
        elif fusion == "gated":
            self.fusion = GatedFusionHead(image_hidden, text_hidden, fusion_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")
        self.classifier = make_classifier(fusion_dim, num_classes)
        if freeze_backbones:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def extract_features(self, pixel_values, input_ids, attention_mask):
        image_tokens = self.image_encoder(pixel_values)
        text_tokens = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.fusion(image_tokens, text_tokens, attention_mask)

    def forward(self, pixel_values, input_ids, attention_mask, return_features: bool = False):
        fused_cls = self.extract_features(pixel_values, input_ids, attention_mask)
        logits = self.classifier(fused_cls)
        if return_features:
            return logits, fused_cls
        return logits


class TextImageFusionDANNKnownOnly(TextImageFusionClosedSet):
    def __init__(self, image_encoder: nn.Module, text_model_name: str, num_classes: int, fusion: Literal["cross_attention", "gated"], fusion_dim: int = 256, num_heads: int = 4, freeze_backbones: bool = False):
        super().__init__(image_encoder, text_model_name, num_classes, fusion, fusion_dim, num_heads, freeze_backbones)
        self.domain_classifier = make_classifier(fusion_dim, 2)

    def forward(self, pixel_values, input_ids, attention_mask, dann_lambda: float = 0.0):
        fused_cls = self.extract_features(pixel_values, input_ids, attention_mask)
        class_logits = self.classifier(fused_cls)
        domain_logits = self.domain_classifier(grad_reverse(fused_cls, dann_lambda))
        return class_logits, domain_logits, fused_cls


class MobileViTCrossAttentionFusion(TextImageFusionClosedSet):
    def __init__(self, image_model_name: str, text_model_name: str, num_classes: int, fusion_dim: int = 256, num_heads: int = 4, freeze_backbones: bool = False):
        super().__init__(MobileViTAdapter(image_model_name), text_model_name, num_classes, "cross_attention", fusion_dim, num_heads, freeze_backbones)


class MobileViTGatedFusion(TextImageFusionClosedSet):
    def __init__(self, image_model_name: str, text_model_name: str, num_classes: int, fusion_dim: int = 256, num_heads: int = 4, freeze_backbones: bool = False):
        super().__init__(MobileViTAdapter(image_model_name), text_model_name, num_classes, "gated", fusion_dim, num_heads, freeze_backbones)


class ResNet50CrossAttentionFusion(TextImageFusionClosedSet):
    def __init__(self, image_model_name: str, text_model_name: str, num_classes: int, fusion_dim: int = 256, num_heads: int = 4, freeze_backbones: bool = False):
        super().__init__(ResNet50Adapter(image_model_name), text_model_name, num_classes, "cross_attention", fusion_dim, num_heads, freeze_backbones)


class ResNet50GatedFusion(TextImageFusionClosedSet):
    def __init__(self, image_model_name: str, text_model_name: str, num_classes: int, fusion_dim: int = 256, num_heads: int = 4, freeze_backbones: bool = False):
        super().__init__(ResNet50Adapter(image_model_name), text_model_name, num_classes, "gated", fusion_dim, num_heads, freeze_backbones)


class MobileViTCrossAttentionDANNKnownOnly(TextImageFusionDANNKnownOnly):
    def __init__(self, image_model_name: str, text_model_name: str, num_classes: int, fusion_dim: int = 256, num_heads: int = 4, freeze_backbones: bool = False):
        super().__init__(MobileViTAdapter(image_model_name), text_model_name, num_classes, "cross_attention", fusion_dim, num_heads, freeze_backbones)


class MobileViTGatedDANNKnownOnly(TextImageFusionDANNKnownOnly):
    def __init__(self, image_model_name: str, text_model_name: str, num_classes: int, fusion_dim: int = 256, num_heads: int = 4, freeze_backbones: bool = False):
        super().__init__(MobileViTAdapter(image_model_name), text_model_name, num_classes, "gated", fusion_dim, num_heads, freeze_backbones)


class ResNet50CrossAttentionDANNKnownOnly(TextImageFusionDANNKnownOnly):
    def __init__(self, image_model_name: str, text_model_name: str, num_classes: int, fusion_dim: int = 256, num_heads: int = 4, freeze_backbones: bool = False):
        super().__init__(ResNet50Adapter(image_model_name), text_model_name, num_classes, "cross_attention", fusion_dim, num_heads, freeze_backbones)


class ResNet50GatedDANNKnownOnly(TextImageFusionDANNKnownOnly):
    def __init__(self, image_model_name: str, text_model_name: str, num_classes: int, fusion_dim: int = 256, num_heads: int = 4, freeze_backbones: bool = False):
        super().__init__(ResNet50Adapter(image_model_name), text_model_name, num_classes, "gated", fusion_dim, num_heads, freeze_backbones)


class ImageOnlyClassifier(nn.Module):
    def __init__(self, image_model_name: str, num_classes: int, fusion_dim: int = 256):
        super().__init__()
        self.image_encoder = FeatureMapAdapter(image_model_name)
        with torch.no_grad():
            hidden = self.image_encoder(torch.randn(1, 3, 224, 224)).shape[-1]
        self.proj = nn.Sequential(nn.Linear(hidden, fusion_dim), nn.LayerNorm(fusion_dim))
        self.classifier = make_classifier(fusion_dim, num_classes)

    def forward(self, pixel_values, input_ids=None, attention_mask=None, return_features: bool = False):
        tokens = self.image_encoder(pixel_values)
        features = self.proj(tokens[:, 0, :])
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


MODEL_REGISTRY = {
    "mobilevit_cross_attention": MobileViTCrossAttentionFusion,
    "mobilevit_gated": MobileViTGatedFusion,
    "resnet50_cross_attention": ResNet50CrossAttentionFusion,
    "resnet50_gated": ResNet50GatedFusion,
}

DANN_MODEL_REGISTRY = {
    "mobilevit_cross_attention": MobileViTCrossAttentionDANNKnownOnly,
    "mobilevit_gated": MobileViTGatedDANNKnownOnly,
    "resnet50_cross_attention": ResNet50CrossAttentionDANNKnownOnly,
    "resnet50_gated": ResNet50GatedDANNKnownOnly,
}


def build_model(model_family: str, image_model_name: str, text_model_name: str, num_classes: int, fusion_dim: int = 256, num_heads: int = 4, freeze_backbones: bool = False):
    if model_family == "image_only":
        return ImageOnlyClassifier(image_model_name, num_classes, fusion_dim)
    cls = MODEL_REGISTRY[model_family]
    return cls(image_model_name, text_model_name, num_classes, fusion_dim, num_heads, freeze_backbones)


def build_dann_model(model_family: str, image_model_name: str, text_model_name: str, num_classes: int, fusion_dim: int = 256, num_heads: int = 4, freeze_backbones: bool = False):
    cls = DANN_MODEL_REGISTRY[model_family]
    return cls(image_model_name, text_model_name, num_classes, fusion_dim, num_heads, freeze_backbones)


# Backwards-compatible aliases matching the original notebook naming style.
MobileViTTextFusionClosedSet = MobileViTGatedFusion
MobileViTDANNKnownOnly = MobileViTGatedDANNKnownOnly
ResNet50TextFusionClosedSet = ResNet50GatedFusion
ResNet50DANNKnownOnly = ResNet50GatedDANNKnownOnly
