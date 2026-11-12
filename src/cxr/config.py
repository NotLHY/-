"""Strict experiment configuration and path resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: StrictStr = Field(min_length=1)
    data_root: Path
    output_dir: Path
    # Either one of TORCHVISION_BACKBONES or any timm model id (e.g.
    # "convnext_small.fb_in22k_ft_in1k"). Timm ids are resolved lazily in model.py so a
    # typo surfaces at build time with timm's own error message.
    backbone: StrictStr = Field(default="efficientnet_b0", min_length=1)
    pretrained: StrictBool = True
    pooling: Literal["image_mean", "study_mean", "study_attention"] = "image_mean"
    loss: Literal["bce", "asl"] = "bce"
    image_size: StrictInt = Field(default=224, ge=32, le=2048)
    batch_size: StrictInt = Field(default=4, ge=1, le=512)
    epochs: StrictInt = Field(default=3, ge=1, le=1000)
    learning_rate: StrictFloat = Field(default=0.0003, gt=0, le=1)
    weight_decay: StrictFloat = Field(default=0.0001, ge=0, le=1)
    seed: StrictInt = Field(default=42, ge=0)
    # The patient split is derived from split_seed alone. None keeps the historical
    # behaviour where one seed controlled both the split and the training run.
    split_seed: StrictInt | None = Field(default=None, ge=0)
    device: Literal["cuda", "cpu"] = "cuda"
    num_workers: StrictInt = Field(default=0, ge=0, le=64)
    max_train_views: StrictInt = Field(default=2, ge=1, le=100)
    train_limit: StrictInt | None = Field(default=None, ge=1)
    val_limit: StrictInt | None = Field(default=None, ge=1)
    val_fraction: StrictFloat = Field(default=0.15, gt=0, lt=1)
    validation: Literal["internal", "official"] = "internal"
    # Train on train.csv + valid.csv combined (final-run data maximisation, B1).
    # Forces an internal split because the official valid is absorbed into train.
    merge_valid: StrictBool = False
    max_train_batches: StrictInt | None = Field(default=None, ge=1)
    max_val_batches: StrictInt | None = Field(default=None, ge=1)
    amp: StrictBool = True
    # Pinned host memory speeds up the copy to a dedicated GPU, but on shared or
    # containerised GPUs cudaHostAlloc can stall for over a minute per batch. Set
    # false on those hosts; it costs a few milliseconds per step and nothing else.
    pin_memory: StrictBool = True
    freeze_backbone: StrictBool = False
    patience: StrictInt = Field(default=5, ge=1)
    cpu_threads: StrictInt = Field(default=4, ge=1, le=256)
    # Structure-changing options default to the old behaviour so existing checkpoints resume.
    view_conditioning: StrictBool = False
    orientation_conditioning: StrictBool = False
    # Runtime options; defaults reproduce the original training loop.
    warmup_epochs: StrictInt = Field(default=0, ge=0, le=1000)
    grad_clip: StrictFloat = Field(default=0.0, ge=0.0, le=1.0)
    log_interval: StrictInt = Field(default=1, ge=1, le=100000)
    channels_last: StrictBool = False
    cudnn_benchmark: StrictBool = False
    max_eval_views: StrictInt | None = Field(default=None, ge=1, le=100)
    reweight_views: StrictBool = False
    # Each patient contributes equal total weight instead of each study doing so.
    # Directly targets the test cohort, which is dominated by frequently followed patients.
    patient_balanced: StrictBool = False
    # Keep only patients whose whole-history study count is at least this value.
    # Used to train a specialist on the long follow-up subpopulation. The count comes
    # from the full train CSV, so it never depends on the holdout split or on test.
    min_patient_studies: StrictInt | None = Field(default=None, ge=1)
    # Initialise weights from another run's checkpoint but train from epoch 1 with a
    # fresh optimiser and schedule (a fine-tune, unlike --resume).
    init_from: Path | None = None
    tta: StrictBool = False
    cache_size: StrictInt = Field(default=320, ge=32, le=2048)
    draft: StrictBool = True
    # Local contrast equalisation on the luminance channel; probe only, default off.
    clahe: StrictBool = False
    # "legacy" keeps the historical affine+jitter; "strong" adds scale/erasing for long runs.
    aug_level: Literal["legacy", "strong"] = "legacy"

    def require_supervised_selection(self) -> None:
        """Reject the historical test-derived selection option at supervised entry points."""
        if self.reweight_views:
            raise ValueError("reweight_views=true is forbidden for training and model selection")

    @property
    def effective_split_seed(self) -> int:
        return self.seed if self.split_seed is None else self.split_seed

    @property
    def partial(self) -> bool:
        return any(
            (
                not self.pretrained,
                self.train_limit is not None,
                self.val_limit is not None,
                self.max_train_batches is not None,
                self.max_val_batches is not None,
            )
        )


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object")
    for field in ("data_root", "output_dir", "init_from"):
        if field in raw and isinstance(raw[field], str):
            raw[field] = Path(raw[field])
    config = Config.model_validate(raw)
    base = path.resolve().parent
    update: dict[str, object] = {
        "data_root": (base / config.data_root).resolve(),
        "output_dir": (base / config.output_dir).resolve(),
    }
    if config.init_from is not None:
        update["init_from"] = (base / config.init_from).resolve()
    config = config.model_copy(update=update)
    override = os.environ.get("CXR_DATA_ROOT", "").strip()
    if override:
        # The platform may mount the same CSVs at a different absolute path.
        config = config.model_copy(update={"data_root": Path(override).expanduser().resolve()})
    return config
