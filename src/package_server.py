"""Build a checksum-verified source/weight handoff archive without copying the 55GB dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import tempfile
import time
from pathlib import Path

from server.manifest import FileEntry, TransferManifest


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_files(root: Path, include_resume: bool) -> list[Path]:
    """Use explicit input roots so caches, raw data and credentials cannot leak into a bundle."""
    paths = {root / name for name in ("README.md", "pyproject.toml", ".python-version", ".dockerignore")}
    for name in ("src", "configs", "docs", "reports", "tools"):
        folder = root / name
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix in {".py", ".sh", ".ps1", ".txt", ".md", ".json", ".csv", ".sha256"} or path.name == "Dockerfile":
                paths.add(path)
    weight = root / ".cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth"
    if not weight.is_file() or digest(weight) != "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934":
        raise ValueError("Required official ImageNet weight is absent or has an invalid SHA256")
    paths.add(weight)
    metadata = {
        "best.pt", "config.json", "environment.json", "split_manifest.json", "best_metrics.json",
        "metrics.jsonl", "validation_predictions.csv", "evaluation_valid.json",
        "evaluation_valid_predictions.csv", "submission.csv", "latency.json", "latency.csv",
    }
    if include_resume:
        metadata.add("last.pt")
    for name in ("local_image_mean_224", "local_study_mean_224", "local_full_asl"):
        run = root / "runs" / name
        if not (run / "best.pt").is_file():
            raise FileNotFoundError(f"Missing reference model: {run / 'best.pt'}")
        paths.update(run / item for item in metadata if (run / item).is_file())
    for path in paths:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe/missing transfer input: {path}")
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def verify_archive(path: Path, manifest: TransferManifest) -> None:
    """Read archive members without extracting and compare their actual bytes to the manifest."""
    with tarfile.open(path, "r:gz") as archive:
        names = set(archive.getnames())
        expected = {f"cxr_project/{entry.path}" for entry in manifest.files}
        expected.add("cxr_project/TRANSFER_MANIFEST.json")
        if names != expected:
            raise ValueError("Archive members differ from the transfer manifest")
        for entry in manifest.files:
            handle = archive.extractfile(f"cxr_project/{entry.path}")
            if handle is None:
                raise ValueError(f"Non-file member: {entry.path}")
            with handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != entry.sha256:
                raise ValueError(f"Input changed during packaging: {entry.path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("transfer/cxr_server_code.tar.gz"))
    parser.add_argument("--include-resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Archive already exists; choose a new output filename: {output}")
    paths = selected_files(root, args.include_resume)
    manifest = TransferManifest(
        created_at_ms=time.time_ns() // 1_000_000,
        files=[FileEntry(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
                         sha256=digest(path)) for path in paths],
        note="Dataset ZIP is separate. Verify before editing configs; last.pt excluded unless requested.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".cxr-bundle-", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=3) as archive:
            for path in paths:
                archive.add(path, arcname="cxr_project/" + path.relative_to(root).as_posix(), recursive=False)
            payload = manifest.model_dump_json(indent=2).encode("utf-8")
            member = tarfile.TarInfo("cxr_project/TRANSFER_MANIFEST.json")
            member.size = len(payload)
            member.mtime = manifest.created_at_ms // 1000
            archive.addfile(member, io.BytesIO(payload))
        verify_archive(temporary, manifest)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    checksum = digest(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        checksum + "  " + output.name + "\n", encoding="ascii",
    )
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size,
                      "files": len(paths), "sha256": checksum, "dataset_included": False}, indent=2))


if __name__ == "__main__":
    main()
