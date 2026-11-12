"""Validate the allocated server runtime without installing or changing GPU drivers."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import torch
import torchvision


def inspect(core_only: bool = False) -> dict[str, object]:
    """Fail explicitly when the runtime differs from the competition baseline."""
    expected = {
        "python": (platform.python_version_tuple()[:2], ("3", "12")),
        "torch": (torch.__version__.split("+")[0], "2.10.0"),
        "torchvision": (torchvision.__version__.split("+")[0], "0.25.0"),
        "cuda_runtime": (torch.version.cuda, "12.8"),
    }
    mismatches = [f"{key}: got {got}, expected {want}" for key, (got, want) in expected.items()
                  if got != want]
    cudnn = torch.backends.cudnn.version()
    if cudnn is None or cudnn // 10000 != 9:
        mismatches.append(f"cuDNN: got {cudnn}, expected major version 9")
    if not torch.cuda.is_available():
        mismatches.append("CUDA GPU unavailable; check the platform GPU allocation")
    if mismatches:
        raise RuntimeError("Runtime mismatch; select the official image. " + "; ".join(mismatches))
    sample = torch.randn(16, 16, device="cuda")
    if not torch.isfinite(sample @ sample).all():
        raise RuntimeError("CUDA matrix operation produced non-finite values")
    packages = {}
    if not core_only:
        for name in ("pydantic", "scikit-learn", "numpy", "pillow"):
            packages[name] = importlib.metadata.version(name)
        import pydantic

        if int(pydantic.__version__.split(".")[0]) != 2:
            raise RuntimeError("Pydantic 2 is required; run setup_native.sh")
    properties = torch.cuda.get_device_properties(0)
    return {
        "timestamp_ms": time.time_ns() // 1_000_000,
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": cudnn, "gpu": properties.name,
        "gpu_memory_bytes": properties.total_memory, "packages": packages,
        "note": "A successful local check is not a T4 latency certification.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.core_only)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
