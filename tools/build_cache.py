"""Build the lossless 320px L-mode cache used by fast repeated experiments.

The cache stores only deterministic geometry: EXIF transpose, grayscale conversion, and
aspect-preserving pad-resize. Augmentation, RGB expansion, and normalization stay online so
every experiment still sees fresh random crops/jitter and the same cache is reusable at
224/288/320.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr.data import _pad_resize  # noqa: E402
from cxr.records import read_available  # noqa: E402

MANIFEST_VERSION = 1


def _cache_id(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _worker(task: tuple[str, str, int, bool, int, bool]) -> tuple[str, int, bool]:
    source_name, destination_name, size, overwrite, compress_level, draft = task
    source = Path(source_name)
    destination = Path(destination_name)
    if destination.is_file() and destination.stat().st_size > 0 and not overwrite:
        return str(destination), int(destination.stat().st_size), True
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        is_jpeg = opened.format == "JPEG"
        if draft and is_jpeg:
            opened.draft(opened.mode, (size, size))
        image = ImageOps.exif_transpose(opened).convert("L")
    image = _pad_resize(image, size)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        image.save(temporary, format="PNG", compress_level=compress_level, optimize=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination), int(destination.stat().st_size), False


def _verify(output: Path, samples: list[tuple[Path, Path]], size: int, draft: bool) -> None:
    if not samples:
        return
    random.Random(20261029).shuffle(samples)
    mismatches = 0
    for source, destination in samples:
        with Image.open(source) as opened:
            is_jpeg = opened.format == "JPEG"
            if draft and is_jpeg:
                opened.draft(opened.mode, (size, size))
            expected = np.asarray(
                _pad_resize(ImageOps.exif_transpose(opened).convert("L"), size), dtype=np.uint8
            )
        with Image.open(destination) as cached:
            actual = np.asarray(cached.convert("L"), dtype=np.uint8)
        if expected.shape != actual.shape or not np.array_equal(expected, actual):
            mismatches += 1
    if mismatches:
        raise SystemExit(f"Cache verification failed for {mismatches}/{len(samples)} samples")
    print(f"verified {len(samples)} samples: pixel-exact", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="e.g. cache/320")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--compress-level", type=int, default=1)
    parser.add_argument("--limit", type=int, help="smoke-test only; marks the manifest partial")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-draft", action="store_true")
    parser.add_argument("--serial", action="store_true", help="single process; useful in sandboxes")
    parser.add_argument("--verify", type=int, default=50)
    args = parser.parse_args()
    if not 32 <= args.size <= 2048:
        parser.error("--size must be in 32..2048")
    if not 1 <= args.workers <= 64:
        parser.error("--workers must be in 1..64")
    if not 0 <= args.compress_level <= 9:
        parser.error("--compress-level must be in 0..9")
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    draft = not args.no_draft
    data = read_available(data_root)

    tasks: list[tuple[str, str, int, bool, int, bool]] = []
    pairs: list[tuple[Path, Path]] = []
    split_counts: dict[str, int] = {}
    for split, part in data.items():
        count = 0
        for study in part.studies:
            for image in study.images:
                relative = image.path.relative_to(data_root)
                destination = output / relative.with_suffix(".png")
                tasks.append(
                    (
                        str(image.path),
                        str(destination),
                        args.size,
                        args.overwrite,
                        args.compress_level,
                        draft,
                    )
                )
                pairs.append((image.path, destination))
                count += 1
        split_counts[split] = count
    if args.limit is not None:
        tasks = tasks[: args.limit]
        pairs = pairs[: args.limit]
        split_counts = {split: min(count, args.limit) for split, count in split_counts.items()}
    total = len(tasks)
    if not total:
        raise SystemExit("No images found")
    print(f"building {total} images at {args.size}px into {output}", flush=True)

    started = time.perf_counter()
    built = skipped = 0

    def consume(results: object) -> None:
        nonlocal built, skipped
        for index, (_, _, existed) in enumerate(results, start=1):  # type: ignore[union-attr]
            built += 0 if existed else 1
            skipped += 1 if existed else 0
            if index % 500 == 0 or index == total:
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed else 0
                remaining = (total - index) / rate if rate else 0
                print(
                    f"{index}/{total} built={built} skipped={skipped} "
                    f"{rate:.1f} img/s eta={remaining:.0f}s",
                    flush=True,
                )

    if args.serial or args.workers == 1:
        consume((_worker(task) for task in tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_worker, task) for task in tasks]
            consume((future.result() for future in as_completed(futures)))

    total_bytes = sum(path.stat().st_size for _, path in pairs if path.is_file())
    manifest = {
        "version": MANIFEST_VERSION,
        "size": args.size,
        "mode": "L",
        "resample": "BILINEAR",
        "pad_fill": 0,
        "draft": draft,
        "compress_level": args.compress_level,
        "data_root": str(data_root),
        "source_csv_sha256": {split: part.sha256 for split, part in data.items()},
        "splits": split_counts,
        "images": total,
        "bytes": total_bytes,
        "partial": args.limit is not None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "bytes", "data_root"}
    }
    manifest["cache_id"] = _cache_id(identity)
    _write_json(output / "manifest.json", manifest)

    if args.verify > 0:
        _verify(output, pairs[: args.verify], args.size, draft)
    print(
        f"done: {output} images={total} bytes={total_bytes / 1e9:.2f}GB "
        f"cache_id={manifest['cache_id'][:16]}...",
        flush=True,
    )


if __name__ == "__main__":
    main()
