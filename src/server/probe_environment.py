"""Inventory a candidate platform image before deciding whether it can run this project."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import typing


def main() -> None:
    versions = {}
    errors = []
    for name in ("torch", "torchvision", "numpy", "pydantic", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    result: dict[str, object] = {
        "python": platform.python_version(), "packages": versions,
        "has_typing_NotRequired": hasattr(typing, "NotRequired"),
        "has_hashlib_file_digest": hasattr(hashlib, "file_digest"),
    }
    try:
        import torch
        import torchvision

        result.update(cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      cuda_available=torch.cuda.is_available(),
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      imported_torchvision=torchvision.__version__)
    except (ImportError, RuntimeError, OSError) as error:
        errors.append(f"Framework import failed: {type(error).__name__}: {error}")
    result["errors"] = errors
    result["note"] = "Inventory only; this does not qualify a non-official training image."
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
