"""Checkpoint evaluation, export, audit, and local latency benchmark."""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import load_checkpoint, write_json, write_predictions
from .config import Config
from .data import (
    ORIENT_CODE,
    VIEW_CODE,
    StudyDataset,
    cache_root_from_env,
    cache_size_from_env,
    collate_studies,
    transform_image,
)
from .logits import (
    FLIP_DIMS,
    LogitCollector,
    attach_records,
    check_dump_against_scores,
    parse_flips,
    write_dump,
)
from .metrics import Metrics
from .holdout import Holdout
from .records import SplitRecords, StudyRecord, read_available
from .runtime import device_for, score
from .splits import manifest_studies


def audit(config: Config, split: str | None = None) -> dict[str, object]:
    data = read_available(config.data_root)
    if split is not None and split not in data:
        raise ValueError(f"Missing {split}.csv")
    selected = {split: data[split]} if split else data
    counts: dict[str, object] = {}
    for name, part in selected.items():
        counts[name] = {
            "studies": len(part.studies),
            "images": sum(len(study.images) for study in part.studies),
            "subjects": len({study.subject_id for study in part.studies}),
            "label_positives": [
                sum(
                    int(study.labels is not None and label in study.labels)
                    for study in part.studies
                )
                for label in range(1, 11)
            ]
            if name != "test"
            else None,
            "csv_sha256": part.sha256,
        }
    report = {
        "splits": counts,
        "overlap": {"patients": 0, "studies": 0},
        "label_note": (
            "Classes absent from a study label are treated as operational negatives; "
            "CSV labels may be incomplete."
        ),
    }
    write_json(config.output_dir / "audit.json", report)
    return report


def _studies_for(
    config: Config, data: dict[str, SplitRecords], split: str | None, checkpoint: Path
) -> tuple[str, list[StudyRecord]]:
    if split is not None:
        if split not in data:
            raise ValueError(f"Missing {split}.csv")
        return split, list(data[split].studies)
    manifest_path = checkpoint.parent / "split_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Default evaluation requires a saved split_manifest.json; specify --split")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["csv_sha256"] != {name: item.sha256 for name, item in data.items()}:
        raise ValueError("CSV hashes differ from saved split manifest")
    if manifest["validation"] == "none":
        raise ValueError("Fixed full-data training has no independent validation scope")
    name = {"internal": "internal", "holdout": "dev", "official": "valid"}[
        manifest["validation"]
    ]
    return name, manifest_studies(data, manifest)


def _checkpoint_partial(state: dict[str, Any]) -> bool:
    saved = state["config"]
    return not saved["pretrained"] or any(
        saved[key] is not None
        for key in ("train_limit", "val_limit", "max_train_batches", "max_val_batches")
    )


def _saved_scope(studies: list[StudyRecord], state: dict[str, Any]) -> list[StudyRecord]:
    ids = state.get("validation_study_ids")
    if ids is None:
        ids = state.get("best_validation_study_ids")
    if ids is None:
        raise ValueError("Checkpoint lacks saved validation scope")
    selected = {study.study_id: study for study in studies}
    if len(ids) != len(set(ids)) or any(key not in selected for key in ids):
        raise ValueError("Saved validation scope differs from split manifest")
    return [selected[key] for key in ids]


def evaluate(
    config: Config,
    checkpoint: Path,
    split: str | None,
    dump_logits: Path | None = None,
    dump_flips: str = "none",
    holdout: Holdout | None = None,
) -> Metrics:
    config.require_supervised_selection()
    if split == "test":
        raise ValueError("Test set has no labels; use predict")
    device = device_for(config)
    data = read_available(config.data_root, splits=("train", "valid"))
    name, studies = _studies_for(config, data, split, checkpoint)
    if name == "test":
        raise ValueError("Test set has no labels; use predict")
    if holdout is not None:
        if name != "train":
            raise ValueError("A holdout manifest scores train studies; pass --split train")
        studies = holdout.select(studies)
        name = "dev"
    model, state = load_checkpoint(checkpoint, config, device)
    if split is None:
        studies = _saved_scope(studies, state)
    collector = LogitCollector(parse_flips(dump_flips)) if dump_logits is not None else None
    metrics, probabilities = score(model, studies, config, device, collector=collector)
    metrics["split"] = name
    metrics["checkpoint"] = str(checkpoint.resolve())
    metrics["partial_smoke"] = config.partial or _checkpoint_partial(state)
    metrics["scope"] = "checkpoint_validation" if split is None else "full_split"
    write_json(config.output_dir / f"evaluation_{name}.json", metrics)
    write_predictions(
        config.output_dir / f"evaluation_{name}_predictions.csv", studies, probabilities
    )
    if dump_logits is not None and collector is not None:
        frames = attach_records(collector.frames(), studies, config.max_eval_views)
        error = check_dump_against_scores(frames, probabilities, config.pooling)
        write_dump(
            dump_logits,
            frames,
            {
                "split": name,
                "scope": metrics["scope"],
                "pooling": config.pooling,
                "backbone": config.backbone,
                "image_size": config.image_size,
                "max_eval_views": config.max_eval_views,
                "flips": list(collector.flips),
                "checkpoint": str(checkpoint.resolve()),
                "studies": len(studies),
                "images": frames.image_count,
                "reconstruction_error": error,
                "macro_auc": metrics["macro_auc"],
            },
        )
        print(f"[dump] {dump_logits} images={frames.image_count} flips={collector.flips}")
    return metrics


def predict(
    config: Config,
    checkpoint: Path,
    split: str,
    output: Path,
    include_subject: bool = False,
    threshold: float | None = None,
    dump_logits: Path | None = None,
    dump_flips: str = "none",
) -> dict[str, object]:
    if split != "test":
        raise ValueError("Prediction export is restricted to test split")
    if threshold is not None and not (0 <= threshold <= 1):
        raise ValueError("Threshold must be in [0,1]")
    device = device_for(config)
    data = read_available(config.data_root)
    studies = list(data["test"].studies)
    model, state = load_checkpoint(checkpoint, config, device)
    dataset = StudyDataset(
        studies,
        config.image_size,
        cache_root=cache_root_from_env(),
        cache_size=cache_size_from_env(config.cache_size),
        max_eval_views=config.max_eval_views,
        data_root=config.data_root,
        draft=config.draft,
        clahe=config.clahe,
        aug_level=config.aug_level,
    )
    collector = LogitCollector(parse_flips(dump_flips)) if dump_logits is not None else None
    if collector is not None and config.tta and "h" not in collector.flips:
        raise ValueError("config.tta needs the 'h' flip in --dump-flips")
    model.eval()
    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(dataset), config.batch_size):
            pixels, mask, view_codes, orientation_codes, _, batch_studies, _ = collate_studies(
                [
                    dataset[index]
                    for index in range(start, min(start + config.batch_size, len(dataset)))
                ]
            )
            if collector is not None:
                mask_rows = np.array(mask, copy=True)
                view_rows = np.array(view_codes, copy=True)
                orientation_rows = np.array(orientation_codes, copy=True)
            pixels = pixels.to(device)
            mask = mask.to(device)
            view_codes = view_codes.to(device)
            orientation_codes = orientation_codes.to(device)
            weight_rows: np.ndarray | None = None
            per_flip: dict[str, np.ndarray] = {}
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda" and config.amp,
            ):
                if collector is None:
                    _, scores = model(pixels, mask, view_codes, orientation_codes)
                    if config.tta:
                        flipped = torch.flip(pixels, dims=[-1])
                        _, flipped_scores = model(flipped, mask, view_codes, orientation_codes)
                        scores = (scores + flipped_scores) / 2
                else:
                    _, scores, view_logits, weights = model.forward_parts(
                        pixels, mask, view_codes, orientation_codes
                    )
                    per_flip["none"] = view_logits.float().cpu().numpy()
                    weight_rows = weights.float().cpu().numpy()
                    for flip in collector.extra:
                        flipped = torch.flip(pixels, dims=FLIP_DIMS[flip])
                        _, flip_scores, flip_logits, _ = model.forward_parts(
                            flipped, mask, view_codes, orientation_codes
                        )
                        per_flip[flip] = flip_logits.float().cpu().numpy()
                        if config.tta and flip == "h":
                            scores = (scores + flip_scores) / 2
            if collector is not None and weight_rows is not None:
                collector.add(
                    list(batch_studies),
                    mask_rows,
                    view_rows,
                    orientation_rows,
                    weight_rows,
                    per_flip,
                )
            all_scores.append(scores.float().cpu().numpy())
    probabilities = np.concatenate(all_scores)
    write_predictions(output, studies, probabilities, include_subject, threshold)
    if dump_logits is not None and collector is not None:
        frames = attach_records(collector.frames(), studies, config.max_eval_views)
        used = ("none", "h") if config.tta else ("none",)
        error = check_dump_against_scores(frames, probabilities, config.pooling, used)
        write_dump(
            dump_logits,
            frames,
            {
                "split": split,
                "pooling": config.pooling,
                "backbone": config.backbone,
                "image_size": config.image_size,
                "max_eval_views": config.max_eval_views,
                "flips": list(collector.flips),
                "config_tta": config.tta,
                "checkpoint": str(checkpoint.resolve()),
                "studies": len(studies),
                "images": frames.image_count,
                "reconstruction_error": error,
                "submission": str(output.resolve()),
            },
        )
        print(f"[dump] {dump_logits} images={frames.image_count} flips={collector.flips}")
    return {
        "output": str(output.resolve()),
        "studies": len(studies),
        "rows": sum(1 for row in probabilities.flat if threshold is None or row >= threshold),
        "partial_smoke": config.partial or _checkpoint_partial(state),
    }


def benchmark(config: Config, checkpoint: Path, repeats: int) -> dict[str, object]:
    if repeats < 1:
        raise ValueError("--repeats must be positive")
    device = device_for(config)
    data = read_available(config.data_root)
    part = next((data[name] for name in ("test", "valid", "train") if name in data), None)
    if part is None:
        raise ValueError("No data split available")
    sample = part.studies[0].images[0]
    path = str(sample.path)
    model, state = load_checkpoint(checkpoint, config, device)
    model.eval()

    def one() -> None:
        image = (
            transform_image(path, config.image_size, False, draft=config.draft)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        mask = torch.ones((1, 1), dtype=torch.bool)
        views = torch.tensor([[VIEW_CODE.get(sample.view.strip().upper(), 0)]])
        orientations = torch.tensor([[ORIENT_CODE.get(sample.orientation.strip().upper(), 0)]])
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda" and config.amp,
            ),
        ):
            _, probability = model(
                image.to(device),
                mask.to(device),
                views.to(device),
                orientations.to(device),
            )
        probability.float().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(min(3, repeats)):
        one()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        one()
        durations.append((time.perf_counter() - start) * 1000)
    result = {
        "hardware": torch.cuda.get_device_name()
        if device.type == "cuda"
        else platform.processor() or platform.machine(),
        "device": str(device),
        "timing_scope": (
            "JPEG decode, full-image resize/pad, normalize, H2D, forward, sigmoid, "
            "D2H and synchronization; excludes model weight load"
        ),
        "median_ms": statistics.median(durations),
        "p95_ms": float(np.percentile(durations, 95)),
        "repeats": repeats,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "weights_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
        ),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "partial_smoke": config.partial or _checkpoint_partial(state),
    }
    write_json(config.output_dir / "benchmark.json", result)
    return result
