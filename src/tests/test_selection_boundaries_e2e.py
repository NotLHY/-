"""Exercise data isolation and complete ten-class comparison via real CLIs."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_e2e import ROOT, build_experiment, invoke


def test_train_evaluate_resume_ignore_test_metadata(tmp_path: Path) -> None:
    """A malformed test file cannot affect supervised training or validation."""
    data, run, config = build_experiment(tmp_path)
    settings = json.loads(config.read_text())
    settings.update(backbone="resnet18", freeze_backbone=True)
    config.write_text(json.dumps(settings))
    (data / "test.csv").write_text("intentionally,invalid\n")
    invoke("train", "--config", str(config))
    manifest = json.loads((run / "split_manifest.json").read_text())
    assert set(manifest["csv_sha256"]) == {"train", "valid"}
    assert manifest.get("test_view_weights") is None
    invoke("evaluate", "--config", str(config), "--checkpoint", str(run / "best.pt"))
    (data / "test.csv").write_text("changed,still,invalid\n")
    settings["epochs"] = 2
    config.write_text(json.dumps(settings))
    invoke("train", "--config", str(config), "--resume", str(run / "last.pt"))
    history = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in history] == [1, 2]
    assert all(row["selection_metric"] == "macro_auc" for row in history)


def test_training_rejects_test_derived_selection(tmp_path: Path) -> None:
    """Unsafe historical configuration must fail rather than silently select weights."""
    _, run, config = build_experiment(tmp_path)
    settings = json.loads(config.read_text())
    settings.update(backbone="resnet18", freeze_backbone=True, reweight_views=True)
    config.write_text(json.dumps(settings))
    result = invoke("train", "--config", str(config), success=False)
    assert "reweight_views" in result.stderr
    assert not (run / "last.pt").exists()


def _comparison(tmp_path: Path, invalid: str = "") -> subprocess.CompletedProcess[str]:
    data, _, _ = build_experiment(tmp_path)
    truth = {}
    with (data / "valid.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            truth[row["Study_id"]] = int(row["Predict_class"])
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Study_id", "Predict_class", "Probability"])
        for study, target in truth.items():
            for label in range(1, 11):
                if invalid == "missing" and label == 10:
                    continue
                value = 0.9 if label == target else 0.1
                if invalid == "range":
                    value = 1.1
                writer.writerow([study, label, value])
        if invalid == "duplicate":
            writer.writerow([next(iter(truth)), 1, 0.8])
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/compare_models.py"),
            "--valid",
            str(data / "valid.csv"),
            "--baseline",
            str(predictions),
            "--asl",
            str(predictions),
            "--output",
            str(tmp_path / "comparison.json"),
            "--reps",
            "40",
            "--seed",
            "42",
        ],
        cwd=ROOT,
        env=dict(os.environ, OMP_NUM_THREADS="2"),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_comparison_reports_undefined_resamples(tmp_path: Path) -> None:
    """All ten classes exist globally, but missing-class resamples are not ten-class AUCs."""
    result = _comparison(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["baseline"]["macro"] == pytest.approx(1.0)
    bootstrap = report["bootstrap"]
    assert bootstrap["reps_used"] + bootstrap["reps_skipped_missing_class"] == 40
    assert bootstrap["reps_skipped_missing_class"] > 0
    if bootstrap["reps_used"] == 0:
        assert bootstrap["ci95"] is None
    stress = report["stress_ge3_checks_per_patient"]
    assert stress["studies"] == 0
    assert stress["baseline"]["macro"] is None


@pytest.mark.parametrize("invalid", ["duplicate", "range", "missing"])
def test_comparison_rejects_invalid_probabilities(tmp_path: Path, invalid: str) -> None:
    result = _comparison(tmp_path, invalid=invalid)
    assert result.returncode != 0
    message = {"duplicate": "Duplicate", "range": "Probability", "missing": "classes"}[invalid]
    assert message in result.stderr
    assert not (tmp_path / "comparison.json").exists()
