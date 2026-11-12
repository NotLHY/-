"""Run-local code snapshot for CXR training (stdlib only).

Every fresh ``cxr train`` writes ``<run>/code_snapshot/`` so a checkpoint can be
traced back to the exact source tree + config + git state that produced it.
The snapshot is deliberately small (source/config/docs only; no data, weights,
caches or results) and works without git (falls back to a plain file-tree
manifest, which is the normal case on the training server).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INCLUDE_DIRS = ("src", "configs", "docs", "tools", "scripts")
INCLUDE_FILES = ("pyproject.toml", "README.md", ".python-version", ".dockerignore")

SKIP_DIR_NAMES = {
    ".git",
    ".git.upstream",
    ".venv",
    "venv",
    "__pycache__",
    "runs",
    "cache",
    "cxr_first",
    "transfer",
    "logs",
    "dist",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".uv-cache",
    ".work",
    ".cache",
    "reports",
    "_analysis",
}

SKIP_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".onnx", ".engine", ".safetensors",
    ".npz", ".npy", ".pkl", ".pickle", ".bin", ".so", ".dll", ".exe",
    ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff",
    ".pdf", ".mp4", ".mov", ".wav",
}

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_PATCH_BYTES = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _git_info(root: Path) -> dict[str, Any]:
    top = _git(root, "rev-parse", "--show-toplevel")
    if not top:
        return {"available": False}
    info: dict[str, Any] = {"available": True, "repo_root": top.strip()}
    for key, args in (
        ("head", ("rev-parse", "HEAD")),
        ("branch", ("rev-parse", "--abbrev-ref", "HEAD")),
        ("remote", ("remote", "get-url", "origin")),
    ):
        value = _git(root, *args)
        info[key] = value.strip() if value else None
    status = _git(root, "status", "--porcelain", "--", ".")
    info["dirty"] = bool(status and status.strip())
    info["status"] = status or ""
    diff = _git(root, "diff", "HEAD", "--", ".") or ""
    if len(diff) > MAX_PATCH_BYTES:
        diff = diff[:MAX_PATCH_BYTES] + "\n... (truncated)\n"
    info["diff_patch"] = diff
    return info


def _environment(config: Any) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import torch  # noqa: PLC0415

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - 没有 torch 时仍可拍快照
        env["torch"] = None
    try:
        env["config"] = config.model_dump(mode="json")
    except Exception:  # noqa: BLE001
        env["config"] = None
    return env


def _collect(root: Path) -> tuple[dict[str, Path], list[dict[str, str]]]:
    selected: dict[str, Path] = {}
    skipped: list[dict[str, str]] = []

    def add(path: Path) -> None:
        if not path.is_file():
            return
        rel = path.relative_to(root)
        if any(part in SKIP_DIR_NAMES for part in rel.parts[:-1]):
            return
        if path.suffix.lower() in SKIP_SUFFIXES:
            skipped.append({"path": rel.as_posix(), "reason": "binary"})
            return
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            skipped.append({"path": rel.as_posix(), "reason": "too-large"})
            return
        selected[rel.as_posix()] = path

    for name in INCLUDE_DIRS:
        base = root / name
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    add(path)
    for name in INCLUDE_FILES:
        add(root / name)
    return selected, skipped


def _normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def write_code_snapshot(run: Path, config: Any, tag: str = "train") -> dict[str, Any]:
    """Write ``run/code_snapshot`` and return its manifest (raises on failure)."""
    run = Path(run)
    out = run / "code_snapshot"
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    files, skipped = _collect(PROJECT_ROOT)
    if not files:
        raise RuntimeError("no source files selected for snapshot")
    out.mkdir(parents=True, exist_ok=True)

    config_payload = None
    try:
        config_payload = config.model_dump(mode="json")
    except Exception:  # noqa: BLE001
        config_payload = None

    archive_tmp = out / ".snapshot.tar.gz.tmp"
    try:
        with tarfile.open(archive_tmp, "w:gz", compresslevel=6) as archive:
            for rel in sorted(files):
                path = files[rel]
                info = _normalize(archive.gettarinfo(str(path), arcname=f"code/{rel}"))
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            if config_payload is not None:
                payload = json.dumps(config_payload, ensure_ascii=False, indent=2).encode("utf-8")
                info = tarfile.TarInfo("config.json")
                info.size = len(payload)
                info.mtime = 0
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(payload))
        archive_sha = _sha256(archive_tmp)
        os.replace(archive_tmp, out / "snapshot.tar.gz")
    finally:
        archive_tmp.unlink(missing_ok=True)

    manifest: dict[str, Any] = {
        "snapshot_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tag": tag,
        "project_root": str(PROJECT_ROOT),
        "git": _git_info(PROJECT_ROOT),
        "environment": _environment(config),
        "archive": {
            "name": "snapshot.tar.gz",
            "bytes": (out / "snapshot.tar.gz").stat().st_size,
            "sha256": archive_sha,
        },
        "file_count": len(files),
        "files": [
            {"path": rel, "bytes": files[rel].stat().st_size, "sha256": _sha256(files[rel])}
            for rel in sorted(files)
        ],
        "skipped": skipped[:200],
        "skipped_count": len(skipped),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "files.sha256").write_text(
        "".join(f"{item['sha256']}  code/{item['path']}\n" for item in manifest["files"]),
        encoding="utf-8",
    )
    return manifest
