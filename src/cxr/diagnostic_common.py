"""Shared measurement boundaries and persisted-data selection for bounded diagnostics."""

from __future__ import annotations

import io
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import Config
from .data import _pad_resize
from .records import StudyRecord, read_available


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def hardware(device: torch.device) -> dict[str, object]:
    return {
        "device": device.type, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
    }


def summary(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "p95": float(np.percentile(values, 95)), "max": max(values)}


def saved_training_studies(config: Config, checkpoint: Path) -> list[StudyRecord]:
    data = read_available(config.data_root)
    manifest = json.loads((checkpoint.parent / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest["csv_sha256"] != {name: part.sha256 for name, part in data.items()}:
        raise ValueError("CSV hashes differ from training manifest")
    ids = manifest["train_study_ids"]
    if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
        raise ValueError("Invalid persisted training study IDs")
    available = {study.study_id: study for study in data["train"].studies}
    if len(ids) != len(set(ids)) or any(key not in available for key in ids):
        raise ValueError("Training scope differs from current dataset")
    return [available[key] for key in ids]


def input_microbenchmark(studies: list[StudyRecord], size: int, limit: int = 12) -> dict[str, object]:
    paths = [image.path for study in studies for image in study.images][:limit]
    reads, decodes, resize = [], [], []
    for path in paths:
        started = time.perf_counter()
        content = path.read_bytes()
        read_end = time.perf_counter()
        with Image.open(io.BytesIO(content)) as source:
            source.load()
            decode_end = time.perf_counter()
            _pad_resize(source.convert("RGB"), size)
            resize_end = time.perf_counter()
        reads.append((read_end - started) * 1000)
        decodes.append((decode_end - read_end) * 1000)
        resize.append((resize_end - decode_end) * 1000)
    return {"images": len(paths), "file_read_ms": summary(reads),
            "jpeg_decode_ms": summary(decodes), "rgb_resize_pad_ms": summary(resize),
            "scope": "Sequential deterministic input microbenchmark; excludes augmentation/tensor normalization."}
