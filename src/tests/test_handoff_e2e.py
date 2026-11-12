"""Verify a real upload bundle and portable server config through public scripts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from test_e2e import ROOT, build_experiment


def test_server_bundle_contains_only_transfer_inputs(tmp_path: Path) -> None:
    """The archive must carry sources and weights, with verifiable relative file hashes."""
    archive = tmp_path / "cxr-server-code.tar.gz"
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/package_server.py"), "--output", str(archive)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with tarfile.open(archive, "r:gz") as bundle:
        names = bundle.getnames()
        assert "cxr_project/src/server/run.sh" in names
        assert "cxr_project/src/cxr/model.py" in names
        assert "cxr_project/configs/local_image_mean.json" in names
        assert "cxr_project/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth" in names
        assert "cxr_project/runs/local_image_mean_224/best.pt" in names
        assert not any("/cxr_first/" in name for name in names)
        assert not any("/.venv/" in name or "/.work/" in name for name in names)
        assert not any(name.endswith("last.pt") for name in names)
        assert not any(name.endswith(".whl") for name in names)
        member = bundle.extractfile("cxr_project/TRANSFER_MANIFEST.json")
        assert member is not None
        manifest = json.load(member)
        for entry in manifest["files"]:
            handle = bundle.extractfile("cxr_project/" + entry["path"])
            assert handle is not None
            assert hashlib.sha256(handle.read()).hexdigest() == entry["sha256"]
    assert archive.with_suffix(archive.suffix + ".sha256").is_file()


def test_server_configuration_is_portable(tmp_path: Path) -> None:
    """Generated formal runs use all selected training data and the school's four CPU quota."""
    output = tmp_path / "generated"
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/server/configure.py"), "--destination", str(output)],
        cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    files = sorted(output.glob("*.json"))
    assert len(files) == 3
    full = [json.loads(path.read_text()) for path in files if "smoke" not in path.name]
    assert {item["pooling"] for item in full} == {"image_mean", "study_mean"}
    assert all(item["train_limit"] is None and item["val_limit"] is None for item in full)
    assert all(item["cpu_threads"] == 4 and item["num_workers"] == 8 for item in full)
    assert len({item["output_dir"] for item in full}) == 2


def test_data_extraction_recovers_a_missing_image(tmp_path: Path) -> None:
    """CSV presence alone must not make an interrupted extraction appear complete."""
    source, _, _ = build_experiment(tmp_path / "source")
    archive = tmp_path / "data.zip"
    paths = [path for path in source.rglob("*") if path.is_file()]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, "cxr_first/" + path.relative_to(source).as_posix())
    spec = {
        "archive_file": "cxr_first.zip", "archive_bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "uncompressed_bytes": sum(path.stat().st_size for path in paths), "files": len(paths),
        "archive_root": "cxr_first",
        "csv_sha256": {split: hashlib.sha256((source / f"{split}.csv").read_bytes()).hexdigest()
                       for split in ("train", "valid", "test")},
        "counts": {split: {"studies": count, "images": len(list((source / split).rglob("*.jpg")))}
                   for split, count in (("train", 20), ("valid", 10), ("test", 3))},
    }
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    destination = tmp_path / "extracted"
    command = [sys.executable, str(ROOT / "src/server/prepare_data.py"), "--archive", str(archive),
               "--manifest", str(manifest), "--destination", str(destination)]
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    first = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, timeout=30)
    assert first.returncode == 0, first.stderr
    image = next((destination / "cxr_first").rglob("*.jpg"))
    expected = image.read_bytes()
    image.unlink()
    resumed = subprocess.run(command + ["--resume-extract"], cwd=ROOT, env=environment,
                             capture_output=True, timeout=30)
    assert resumed.returncode == 0, resumed.stderr
    assert image.is_file()
    assert image.read_bytes() == expected
