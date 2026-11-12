"""Run artifacts and safe tensor checkpoint boundaries."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import pydantic
import sklearn
import torch
import torchvision

from .config import Config
from .metrics import Metrics
from .model import StudyModel, weight_provenance
from .records import StudyRecord


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    temporary = _temporary(path)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_environment(path: Path, config: Config, device: torch.device) -> None:
    write_json(path, {
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": np.__version__,
        "sklearn": sklearn.__version__, "pillow": PIL.__version__,
        "pydantic": pydantic.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "device": str(device),
        "cuda_device": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "weights": weight_provenance(config), "partial_smoke": config.partial,
    })


def write_predictions(path: Path, studies: list[StudyRecord] | tuple[StudyRecord, ...],
                      probabilities: np.ndarray, include_subject: bool = False,
                      threshold: float | None = None) -> None:
    if probabilities.shape != (len(studies), 10) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability shape or value")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probability outside [0,1]")
    fields = (["Subject_id"] if include_subject else []) + [
        "Study_id", "Predict_class", "Probability",
    ]
    temporary = _temporary(path)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for study, scores in zip(studies, probabilities, strict=True):
                for label, score in enumerate(scores, start=1):
                    if threshold is not None and score < threshold:
                        continue
                    row: dict[str, object] = {"Study_id": study.study_id, "Predict_class": label,
                                              "Probability": float(score)}
                    if include_subject:
                        row = {"Subject_id": study.subject_id, **row}
                    writer.writerow(row)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def make_checkpoint(model: StudyModel, optimizer: torch.optim.Optimizer,
                    scheduler: Any,
                    scaler: torch.amp.GradScaler, epoch: int, best_metrics: Metrics | None,
                    config: Config) -> dict[str, Any]:
    # Torch optimizer/RNG state is a heterogeneous tensor/primitives mapping; Any is confined here.
    numpy_rng = np.random.get_state()
    return {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "epoch": epoch, "best_metrics": best_metrics,
        "config": config.model_dump(mode="json"), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": (
            numpy_rng[0], numpy_rng[1].tolist(), numpy_rng[2], numpy_rng[3], numpy_rng[4],
        ),
    }


def load_checkpoint(
    path: Path, config: Config, device: torch.device,
) -> tuple[StudyModel, dict[str, Any]]:
    # weights_only rejects objects; saved state contains tensors and primitives.
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    saved = state.get("config", {})
    for key in ("backbone", "pooling", "image_size", "loss", "view_conditioning",
                "orientation_conditioning"):
        saved_value = saved.get(key)
        if key in {"view_conditioning", "orientation_conditioning"} and saved_value is None:
            saved_value = False
        if saved_value != getattr(config, key):
            raise ValueError(f"Checkpoint {key} differs from config")
    model = StudyModel(config, load_pretrained=False)
    model.load_state_dict(state["model"])
    return model.to(device), state
