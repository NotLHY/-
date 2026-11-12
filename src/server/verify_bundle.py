"""Verify every transferred file before generating configs or starting training."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from server.manifest import TransferManifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = TransferManifest.model_validate_json((root / "TRANSFER_MANIFEST.json").read_text())
    seen: set[str] = set()
    for entry in manifest.files:
        if entry.path in seen:
            raise ValueError(f"Duplicate manifest path: {entry.path}")
        seen.add(entry.path)
        path = root / entry.path
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe transferred path: {entry.path}")
        if not path.is_file() or path.stat().st_size != entry.bytes:
            raise ValueError(f"Missing/size-mismatched file: {entry.path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry.sha256:
            raise ValueError(f"SHA256 mismatch: {entry.path}")
    print(f"Verified {len(seen)} files; immutable transferred inputs are intact.")


if __name__ == "__main__":
    main()
