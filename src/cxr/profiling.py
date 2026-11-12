"""Measure bounded real-data waits and training compute using disposable in-memory models."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .artifacts import load_checkpoint, write_json
from .config import Config, load_config
from .data import (
    StudyDataset,
    cache_root_from_env,
    cache_size_from_env,
    collate_studies,
)
from .diagnostic_common import hardware, input_microbenchmark, saved_training_studies, summary, synchronize
from .records import StudyRecord
from .runtime import _worker_init, device_for, seed_all


def measure_case(config: Config, checkpoint: Path, studies: list[StudyRecord],
                 device: torch.device, workers: int, batches: int, warmup: int,
                 cached: bool) -> tuple[dict[str, object], list[dict[str, object]]]:
    seed_all(config.seed)
    model, _ = load_checkpoint(checkpoint, config, device)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.amp)
    dataset = StudyDataset(
        studies,
        config.image_size,
        True,
        config.max_train_views,
        config.seed,
        cache_root=cache_root_from_env(),
        cache_size=cache_size_from_env(config.cache_size),
        data_root=config.data_root,
        draft=config.draft,
    )
    options = (
        {"worker_init_fn": _worker_init, "prefetch_factor": 4, "persistent_workers": True}
        if workers > 0
        else {}
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=workers,
        collate_fn=collate_studies,
        pin_memory=device.type == "cuda",
        **options,
    )
    if not cached and len(loader) < batches + warmup:
        raise ValueError("Requested more batches than saved training scope; reduce --batches")
    started = time.perf_counter()
    iterator = iter(loader)
    startup_ms = (time.perf_counter() - started) * 1000
    cached_batch = None
    cached_counts = None
    if cached:
        pixels, mask, views, orientations, labels, _, _ = next(iterator)
        cached_counts = (int(labels.shape[0]), int(mask.sum().item()))
        cached_batch = tuple(
            value.to(device) for value in (pixels, mask, views, orientations, labels)
        )
        synchronize(device)
    name = "cached_batch_compute" if cached else f"data_workers_{workers}"
    rows: list[dict[str, object]] = []
    for step in range(warmup + batches):
        synchronize(device)
        start = time.perf_counter()
        if cached_batch is None:
            pixels, mask, views, orientations, labels, _, _ = next(iterator)
            counts = (int(labels.shape[0]), int(mask.sum().item()))
            loaded = time.perf_counter()
            inputs = tuple(
                value.to(device) for value in (pixels, mask, views, orientations, labels)
            )
            synchronize(device)
            transferred = time.perf_counter()
        else:
            inputs = cached_batch
            counts = cached_counts
            loaded = transferred = start
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=device.type == "cuda" and config.amp):
            loss, _ = model.study_loss(*inputs, config.loss)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite profiling loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        end = time.perf_counter()
        if step == warmup - 1 and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if step >= warmup:
            if counts is None:
                raise RuntimeError("Missing profiling counts")
            rows.append({"case": name, "step": step - warmup + 1,
                         "studies": counts[0], "images": counts[1],
                         "data_wait_ms": (loaded - start) * 1000,
                         "transfer_ms": (transferred - loaded) * 1000,
                         "compute_ms": (end - transferred) * 1000,
                         "total_ms": (end - start) * 1000})
    del iterator
    numeric = {key: summary([float(row[key]) for row in rows])
               for key in ("data_wait_ms", "transfer_ms", "compute_ms", "total_ms")}
    total_seconds = sum(float(row["total_ms"]) for row in rows) / 1000
    result = {"name": name, "workers": workers, "cached_batch": cached,
              "measured_batches": len(rows), "warmup_batches": warmup,
              "startup_ms": startup_ms, "total_ms_mean": numeric["total_ms"]["mean"],
              "stages": numeric,
              "studies_per_second": sum(int(row["studies"]) for row in rows) / total_seconds,
              "images_per_second": sum(int(row["images"]) for row in rows) / total_seconds,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    print(f"{name}: {result['total_ms_mean']:.2f} ms/batch", flush=True)
    return result, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--workers", default="0,2,4")
    args = parser.parse_args()
    workers = [int(value) for value in args.workers.split(",")]
    if not 1 <= args.batches <= 200 or not 0 <= args.warmup <= 20 or not workers:
        parser.error("Use 1..200 batches, 0..20 warmup and a worker list")
    if len(workers) != len(set(workers)) or any(value < 0 or value > 16 for value in workers):
        parser.error("Workers must be distinct values in 0..16")
    config = load_config(args.config)
    device = device_for(config)
    studies = saved_training_studies(config, args.checkpoint)
    cases, rows = [], []
    for workers_count in workers:
        result, records = measure_case(config, args.checkpoint, studies, device, workers_count,
                                       args.batches, args.warmup, False)
        cases.append(result)
        rows.extend(records)
    result, records = measure_case(config, args.checkpoint, studies, device, 0,
                                   args.batches, args.warmup, True)
    cases.append(result)
    rows.extend(records)
    report = {"hardware": hardware(device), "cases": cases,
              "checkpoint": str(args.checkpoint.resolve()), "batch_size": config.batch_size,
              "image_size": config.image_size, "seed": config.seed,
              "input_microbenchmark": input_microbenchmark(studies, config.image_size),
              "methodology": "Data wait surrounds next(iterator). Stages synchronize GPU; warmup/startup excluded. Cached batch is a compute ceiling, not model-quality training.",
              "checkpoints_modified": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "steps.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output_dir / "profile.json", report)


if __name__ == "__main__":
    main()
