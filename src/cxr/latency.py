"""Measure distinct real images through batch-one end-to-end inference on the current GPU."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import torch

from .artifacts import load_checkpoint, write_json
from .config import load_config
from .data import ORIENT_CODE, VIEW_CODE, transform_image
from .diagnostic_common import hardware, summary, synchronize
from .records import read_split
from .runtime import device_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--images", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.images <= 512 or not 0 <= args.warmup <= 50:
        parser.error("Use 1..512 measured images and 0..50 warmup images")
    config = load_config(args.config)
    device = device_for(config)
    images = [image for study in read_split(config.data_root, "test").studies for image in study.images]
    chosen = random.Random(config.seed).sample(images, min(args.images, len(images)))
    model, _ = load_checkpoint(args.checkpoint, config, device)
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for index in range(args.warmup + len(chosen)):
            image = chosen[index % len(chosen)] if index < args.warmup else chosen[index - args.warmup]
            synchronize(device)
            started = time.perf_counter()
            pixels = transform_image(
                str(image.path), config.image_size, False, draft=config.draft
            ).unsqueeze(0).unsqueeze(0)
            mask = torch.ones((1, 1), dtype=torch.bool, device=device)
            views = torch.tensor([[VIEW_CODE.get(image.view.strip().upper(), 0)]], device=device)
            orientations = torch.tensor(
                [[ORIENT_CODE.get(image.orientation.strip().upper(), 0)]], device=device
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda" and config.amp):
                _, probability = model(pixels.to(device), mask, views, orientations)
            scores = probability.float().cpu()
            synchronize(device)
            elapsed = (time.perf_counter() - started) * 1000
            if not torch.isfinite(scores).all():
                raise ValueError("Non-finite latency inference output")
            if index == args.warmup - 1 and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if index >= args.warmup:
                rows.append({"image_id": image.image_id, "view": image.view,
                             "bytes": image.path.stat().st_size, "latency_ms": elapsed})
    timings = summary([float(row["latency_ms"]) for row in rows])
    report = {"hardware": hardware(device), "image_count": len(chosen), "warmup": args.warmup,
              "median_ms": timings["median"], "p95_ms": timings["p95"], "mean_ms": timings["mean"],
              "max_ms": timings["max"], "checkpoint_bytes": args.checkpoint.stat().st_size,
              "weights_bytes": sum(value.numel() * value.element_size() for value in model.state_dict().values()),
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "timing_scope": "Batch1 JPEG decode, canonical full-image preprocess, H2D, forward/sigmoid, D2H/synchronize; model load and CSV audit excluded.",
              "selection": f"Uniform distinct test images; seed {config.seed}; no labels used.",
              "filesystem_cache": "Natural OS cache; no cache flush. Warmup may overlap sampled images.",
              "t4_certified": False}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output_json, report)
    print(f"{len(chosen)} distinct images: median={timings['median']:.2f} ms p95={timings['p95']:.2f} ms")


if __name__ == "__main__":
    main()
