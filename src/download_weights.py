"""Verify manually downloaded official ImageNet weights without making network requests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torchvision.models import EfficientNet_B0_Weights


def main() -> None:
    """Record provenance only after the local file passes the official hash check."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    destination = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    expected_prefix = destination.stem.rsplit("-", 1)[1]
    if not destination.is_file():
        raise FileNotFoundError(f"Download {weights.url} and save it to {destination}")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if not digest.startswith(expected_prefix):
        raise ValueError(f"Official weight hash mismatch: expected {expected_prefix}, got {digest}")
    weights.get_state_dict(progress=False, check_hash=True)
    metadata = {
        "weight_enum": str(weights), "url": weights.url, "sha256": digest,
        "source_dataset": "ImageNet-1K", "bytes": destination.stat().st_size,
    }
    Path("docs/pretraining.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
