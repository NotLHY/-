"""ImageNet encoders, view/orientation conditioning, and study-level pooling."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    densenet121,
    efficientnet_b0,
    resnet18,
)

from .config import Config

WEIGHTS: dict[str, EfficientNet_B0_Weights | ResNet18_Weights | DenseNet121_Weights] = {
    "efficientnet_b0": EfficientNet_B0_Weights.DEFAULT,
    "resnet18": ResNet18_Weights.DEFAULT,
    "densenet121": DenseNet121_Weights.DEFAULT,
}

# Everything else is treated as a timm model id and resolved by timm.create_model.
TORCHVISION_BACKBONES = frozenset(WEIGHTS)


def _timm_checksum(model_id: str) -> str | None:
    """sha256 of the locally cached safetensors, or None when it is not downloaded yet."""
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - provenance must never break training
        return None
    try:
        found = try_to_load_from_cache(f"timm/{model_id}", "model.safetensors")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(found, str):
        return None
    path = Path(found)
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weight_provenance(config: Config) -> dict[str, str | None]:
    if not config.pretrained:
        return {"backbone": config.backbone, "weight_enum": None, "url": None, "checksum": None}
    if config.backbone not in TORCHVISION_BACKBONES:
        return {
            "backbone": config.backbone,
            "weight_enum": f"timm:{config.backbone}",
            "url": f"https://huggingface.co/timm/{config.backbone}",
            "checksum": _timm_checksum(config.backbone),
        }
    weights = WEIGHTS[config.backbone]
    cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    checksum = None
    if cached.is_file():
        with cached.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "backbone": config.backbone,
        "weight_enum": str(weights),
        "url": weights.url,
        "checksum": checksum,
    }


class StudyModel(nn.Module):
    def __init__(self, config: Config, load_pretrained: bool = True) -> None:
        super().__init__()
        # Plain tuple leaves the checkpoint state dictionary backward compatible.
        self.class_loss_weights = (
            tuple(config.class_loss_weights) if config.class_loss_weights is not None else None
        )
        self.encoder: nn.Module
        if config.backbone == "efficientnet_b0":
            network = efficientnet_b0(
                weights=EfficientNet_B0_Weights.DEFAULT
                if config.pretrained and load_pretrained
                else None
            )
            self.encoder = nn.Sequential(network.features, network.avgpool, nn.Flatten())
            dimension = 1280
        elif config.backbone == "densenet121":
            network = densenet121(
                weights=DenseNet121_Weights.DEFAULT
                if config.pretrained and load_pretrained
                else None
            )
            self.encoder = nn.Sequential(
                network.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
            )
            dimension = 1024
        elif config.backbone == "resnet18":
            network = resnet18(
                weights=ResNet18_Weights.DEFAULT if config.pretrained and load_pretrained else None
            )
            self.encoder = nn.Sequential(*list(network.children())[:-1], nn.Flatten())
            dimension = 512
        else:
            # Any other id is a timm model (ConvNeXt, ViT, ...). num_classes=0 makes timm
            # return pooled features. Convolutional families ignore img_size, ViT families
            # need it (plus dynamic_img_size) to accept a resolution other than their default.
            try:
                timm_model = timm.create_model(
                    config.backbone,
                    img_size=config.image_size,
                    dynamic_img_size=True,
                    pretrained=config.pretrained and load_pretrained,
                    num_classes=0,
                )
            except TypeError:
                timm_model = timm.create_model(
                    config.backbone, pretrained=config.pretrained and load_pretrained, num_classes=0
                )
            self.encoder = timm_model
            dimension = int(cast(int, timm_model.num_features))
        self.pooling = config.pooling
        self.frozen = config.freeze_backbone
        self.channels_last = config.channels_last
        if self.frozen:
            self.encoder.requires_grad_(False)
        self.classifier = nn.Linear(dimension, 10)
        self.attention = nn.Linear(dimension, 1) if self.pooling == "study_attention" else None
        # Attention needs a view cue even when head conditioning is disabled.
        self.view_embedding = (
            nn.Embedding(6, dimension) if self.pooling == "study_attention" else None
        )
        self.condition_view_embedding = (
            nn.Embedding(6, dimension)
            if config.view_conditioning and self.pooling != "study_attention"
            else None
        )
        self.orientation_embedding = (
            nn.Embedding(4, dimension) if config.orientation_conditioning else None
        )
        # Parameter-free normalization keeps old checkpoints loadable while stabilising attention.
        self.attention_norm = (
            nn.LayerNorm(dimension, elementwise_affine=False)
            if self.pooling == "study_attention"
            else None
        )
        for module in (
            self.view_embedding, self.condition_view_embedding, self.orientation_embedding
        ):
            if module is not None:
                nn.init.normal_(module.weight, std=0.02)

    def train(self, mode: bool = True) -> StudyModel:
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def _condition(
        self, features: Tensor, view_codes: Tensor, orientation_codes: Tensor
    ) -> Tensor:
        if self.condition_view_embedding is not None:
            features = features + self.condition_view_embedding(view_codes)
        if self.orientation_embedding is not None:
            features = features + self.orientation_embedding(orientation_codes)
        return features

    def _head(
        self, pixels: Tensor, mask: Tensor, view_codes: Tensor, orientation_codes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(logits, probability, view_logits, view_weights)``.

        ``view_logits`` is the classifier applied to every view's conditioned features
        and ``view_weights`` is the pooling weight actually used, so the linear head
        guarantees ``logits == (view_logits * view_weights).sum(dim=1)``. That identity
        is what lets a logit dump rebuild study scores without another forward pass.
        """
        batch, views, channels, height, width = pixels.shape
        flat = pixels.reshape(batch * views, channels, height, width)
        if self.channels_last:
            flat = flat.contiguous(memory_format=torch.channels_last)
        flat_mask = mask.reshape(-1)
        selected = flat[flat_mask]
        encoded = self.encoder(selected)
        features = encoded.new_zeros((batch * views, encoded.shape[-1]))
        features[flat_mask] = encoded
        features = features.reshape(batch, views, -1)
        if self.pooling == "image_mean":
            features = self._condition(features, view_codes, orientation_codes)
            view_logits = self.classifier(features)
            weights = mask.to(view_logits.dtype) / mask.sum(dim=1, keepdim=True)
            logits = (view_logits * weights.unsqueeze(-1)).sum(dim=1)
            probability = (torch.sigmoid(view_logits) * weights.unsqueeze(-1)).sum(dim=1)
            return logits, probability, view_logits, weights
        if self.attention is None:
            features = self._condition(features, view_codes, orientation_codes)
            weights = mask.to(features.dtype) / mask.sum(dim=1, keepdim=True)
        else:
            if self.view_embedding is None:
                raise RuntimeError("Attention view embedding unavailable")
            attended = features + self.view_embedding(view_codes)
            if self.orientation_embedding is not None:
                attended = attended + self.orientation_embedding(orientation_codes)
            if self.attention_norm is not None:
                attended = self.attention_norm(attended)
            weights = torch.softmax(
                self.attention(attended).squeeze(-1).masked_fill(~mask, -torch.inf), dim=1
            )
        pooled = (features * weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(pooled)
        probability = torch.sigmoid(logits)
        return logits, probability, self.classifier(features), weights

    def forward_parts(
        self,
        pixels: Tensor,
        mask: Tensor,
        view_codes: Tensor | None = None,
        orientation_codes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Pooled logits and probability plus the per-view logits and pooled weights."""
        if view_codes is None:
            view_codes = torch.zeros_like(mask, dtype=torch.long)
        if orientation_codes is None:
            orientation_codes = torch.zeros_like(mask, dtype=torch.long)
        return self._head(pixels, mask, view_codes, orientation_codes)

    def forward(
        self,
        pixels: Tensor,
        mask: Tensor,
        view_codes: Tensor | None = None,
        orientation_codes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if view_codes is None:
            view_codes = torch.zeros_like(mask, dtype=torch.long)
        if orientation_codes is None:
            orientation_codes = torch.zeros_like(mask, dtype=torch.long)
        logits, probability, _, _ = self._head(pixels, mask, view_codes, orientation_codes)
        return logits, probability

    def study_loss(
        self,
        pixels: Tensor,
        mask: Tensor,
        view_codes: Tensor,
        orientation_codes: Tensor,
        targets: Tensor,
        loss_name: str,
        sink: Callable[[Tensor, Tensor], None] | None = None,
    ) -> tuple[Tensor, Tensor]:
        logits, probs, view_logits, view_weights = self._head(
            pixels, mask, view_codes, orientation_codes
        )
        if sink is not None:
            sink(view_logits, view_weights)
        if self.pooling == "image_mean":
            per_view = multilabel_loss(
                view_logits, targets.unsqueeze(1).expand_as(view_logits), loss_name
            )
            if self.class_loss_weights is not None:
                weights = per_view.new_tensor(self.class_loss_weights)
                per_view = per_view * (weights / weights.mean())
            per_view = per_view.mean(dim=-1)
            loss = ((per_view * mask).sum(dim=1) / mask.sum(dim=1)).mean()
        else:
            terms = multilabel_loss(logits, targets, loss_name)
            if self.class_loss_weights is not None:
                weights = terms.new_tensor(self.class_loss_weights)
                terms = terms * (weights / weights.mean())
            loss = terms.mean()
        return loss, probs


def multilabel_loss(logits: Tensor, targets: Tensor, name: str) -> Tensor:
    if name == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = torch.sigmoid(logits)
    negative = (1 - positive + 0.05).clamp(max=1)
    positive_loss = targets * F.logsigmoid(logits) * (1 - positive)
    negative_loss = (1 - targets) * torch.log(negative.clamp(min=1e-8)) * (1 - negative).pow(4)
    return -(positive_loss + negative_loss)
