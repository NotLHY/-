"""Verify and extract the original dataset ZIP without deleting or silently overwriting data."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath

from server.manifest import DataManifest


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_csvs(root: Path, manifest: DataManifest) -> None:
    for split, expected in manifest.csv_sha256.items():
        if split not in {"train", "valid", "test"}:
            raise ValueError("Unexpected data split in manifest")
        path = root / f"{split}.csv"
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Dataset CSV missing or changed: {path}")


def existing_matches(path: Path, info: zipfile.ZipInfo) -> bool:
    if not path.is_file() or path.stat().st_size != info.file_size:
        return False
    crc = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF == info.CRC


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--destination", type=Path, default=project)
    parser.add_argument("--manifest", type=Path, default=project / "docs/server-data-manifest.json")
    parser.add_argument("--resume-extract", action="store_true")
    args = parser.parse_args()
    spec = DataManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    data = destination / spec.archive_root
    csvs_present = all((data / f"{split}.csv").is_file() for split in ("train", "valid", "test"))
    images_present = sum(1 for _ in data.rglob("*.jpg")) if data.exists() else 0
    expected_images = sum(part.images for part in spec.counts.values())
    if csvs_present and images_present == expected_images and not args.resume_extract:
        validate_csvs(data, spec)
        print(f"Dataset CSVs match: {data}. Run the audit command to verify every image path.")
        return
    if args.archive is None:
        raise ValueError("Supply --archive or point --destination to an existing dataset parent")
    archive_path = args.archive.resolve()
    if archive_path.stat().st_size != spec.archive_bytes:
        raise ValueError("Dataset ZIP size mismatch; finish/resume upload first")
    print("Verifying the full dataset ZIP SHA256...", flush=True)
    if sha256(archive_path) != spec.sha256:
        raise ValueError("Dataset ZIP SHA256 mismatch; do not extract it")
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        files = [item for item in members if not item.is_dir()]
        if len(files) != spec.files or sum(item.file_size for item in files) != spec.uncompressed_bytes:
            raise ValueError("Dataset archive contents differ from manifest")
        for info in members:
            name = PurePosixPath(info.filename)
            if (name.is_absolute() or ".." in name.parts or not name.parts
                    or name.parts[0] != spec.archive_root or "\\" in info.filename
                    or ":" in info.filename or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            target = destination / info.filename
            if not target.resolve().is_relative_to(destination) or target.is_symlink():
                raise ValueError(f"Unsafe extraction target: {target}")
        existing = sum(item.file_size for item in files
                       if (destination / item.filename).is_file()
                       and (destination / item.filename).stat().st_size == item.file_size)
        needed = spec.uncompressed_bytes - existing + 1024**3
        if shutil.disk_usage(destination).free < needed:
            raise OSError(f"Insufficient free disk: need at least {needed / 1024**3:.1f} GiB")
        completed = 0
        for info in files:
            target = destination / info.filename
            if target.exists():
                if not args.resume_extract:
                    raise FileExistsError(f"Partial dataset exists: {target}; use --resume-extract")
                if existing_matches(target, info):
                    completed += 1
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=".extract-", dir=target.parent)
            os.close(descriptor)
            temporary = Path(name)
            try:
                with archive.open(info) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            completed += 1
            if completed % 1000 == 0:
                print(f"Extracted/verified {completed}/{len(files)} files", flush=True)
    validate_csvs(data, spec)
    print(f"Dataset ready: {data}; original ZIP retained.")


if __name__ == "__main__":
    main()
