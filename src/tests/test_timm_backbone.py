"""Timm backbones must build, forward, backward and round-trip without touching the GPU.

No pytest tmp_path here: the sandbox denies directory enumeration under the temp root,
so the config paths below are inert placeholders (model construction never touches disk).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch

from cxr.config import Config
from cxr.model import TORCHVISION_BACKBONES, StudyModel, weight_provenance

TIMM_ID = "convnext_small.fb_in22k_ft_in1k"
PLACEHOLDER = Path("D:/nonexistent/p6")


def make_config(**overrides: object) -> Config:
    raw: dict[str, object] = {
        "name": "timm_smoke",
        "data_root": PLACEHOLDER / "data",
        "output_dir": PLACEHOLDER / "out",
        "backbone": TIMM_ID,
        "pretrained": False,
        "image_size": 64,
        "batch_size": 1,
        "epochs": 1,
        "device": "cpu",
        "num_workers": 0,
        "cache_size": 320,
    }
    raw.update(overrides)
    return Config.model_validate(raw)


def test_torchvision_default_still_validates() -> None:
    config = make_config(backbone="efficientnet_b0")
    assert config.backbone in TORCHVISION_BACKBONES
    assert weight_provenance(config)["weight_enum"] is None


def test_timm_backbone_forward_backward_and_reload() -> None:
    config = make_config()
    # Provenance is metadata-only: it reads the local cache but never downloads.
    provenance = weight_provenance(make_config(pretrained=True))
    assert provenance["weight_enum"] == f"timm:{TIMM_ID}"
    assert provenance["url"] == f"https://huggingface.co/timm/{TIMM_ID}"

    model = StudyModel(config, load_pretrained=False)
    pixels = torch.randn(1, 2, 3, 64, 64)
    mask = torch.tensor([[True, False]])
    view_codes = torch.zeros_like(mask, dtype=torch.long)
    orientation_codes = torch.zeros_like(mask, dtype=torch.long)
    targets = torch.zeros(1, 10)
    targets[0, 2] = 1.0

    logits, probs = model(pixels, mask, view_codes, orientation_codes)
    assert logits.shape == (1, 10)
    assert probs.shape == (1, 10)
    assert bool(torch.all(probs >= 0)) and bool(torch.all(probs <= 1))

    loss, _ = model.study_loss(pixels, mask, view_codes, orientation_codes, targets, "asl")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "backward produced no gradients"
    assert all(bool(torch.isfinite(g).all()) for g in grads)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    reloaded = StudyModel(config, load_pretrained=False)
    reloaded.load_state_dict(torch.load(buffer, weights_only=True), strict=True)


def test_unknown_backbone_fails_loudly() -> None:
    config = make_config(backbone="definitely_not_a_model")
    with pytest.raises(Exception):
        StudyModel(config, load_pretrained=False)
