"""Exercise diagnostic CLIs on real fixture JPEGs without modifying trained checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from test_e2e import ROOT, build_experiment, invoke


def diagnostic(module: str, *arguments: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="2"),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_diagnostics_leave_training_artifacts_unchanged(tmp_path: Path) -> None:
    """Both diagnostic reports must be bounded and preserve the source checkpoint bytes."""
    _, run, config = build_experiment(tmp_path)
    settings = json.loads(config.read_text(encoding="utf-8"))
    settings.update(backbone="resnet18", freeze_backbone=True)
    config.write_text(json.dumps(settings), encoding="utf-8")
    invoke("train", "--config", str(config))
    checkpoint = run / "best.pt"
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    history_before = (run / "metrics.jsonl").read_bytes()
    output = tmp_path / "profile"
    diagnostic("cxr.profiling", "--config", str(config), "--checkpoint", str(checkpoint),
               "--batches", "2", "--warmup", "1", "--workers", "0", "--output-dir", str(output))
    profile = json.loads((output / "profile.json").read_text(encoding="utf-8"))
    assert profile["hardware"]["device"] == "cpu"
    assert len(profile["cases"]) >= 2
    assert all(case["measured_batches"] == 2 for case in profile["cases"])
    assert all(case["total_ms_mean"] > 0 for case in profile["cases"])
    assert (output / "steps.csv").is_file()
    latency = tmp_path / "latency.json"
    diagnostic("cxr.latency", "--config", str(config), "--checkpoint", str(checkpoint),
               "--images", "3", "--warmup", "1", "--output-json", str(latency))
    measured = json.loads(latency.read_text(encoding="utf-8"))
    assert measured["image_count"] == 3
    assert measured["median_ms"] > 0
    assert measured["p95_ms"] >= measured["median_ms"]
    assert measured["hardware"]["device"] == "cpu"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    assert (run / "metrics.jsonl").read_bytes() == history_before
