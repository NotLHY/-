"""Real CLI regression artifacts for dev selection and fixed full-data training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import torch
from test_e2e import build_experiment, invoke
from test_holdout_dev_e2e import write_holdout


def test_dev_selection_merge_resume(tmp_path: Path) -> None:
    _, run, config = build_experiment(tmp_path)
    settings = json.loads(config.read_text())
    settings.update(
        validation="holdout",
        merge_valid=True,
        backbone="resnet18",
        freeze_backbone=True,
        loss="asl",
        class_loss_weights=[1.5] + [1.0] * 9,
    )
    config.write_text(json.dumps(settings))
    holdout = write_holdout(
        tmp_path / "holdout.json",
        ["subject_train_000", "subject_train_001"],
        ["study_train_000", "study_train_001"],
    )
    invoke("train", "--config", str(config), "--holdout-manifest", str(holdout))
    manifest = json.loads((run / "split_manifest.json").read_text())
    assert len(manifest["train_study_ids"]) == 28
    assert manifest["valid_study_ids"] == ["study_train_000", "study_train_001"]
    assert not set(manifest["train_subject_ids"]) & set(manifest["valid_subject_ids"])
    invoke(
        "train",
        "--config",
        str(config),
        "--holdout-manifest",
        str(holdout),
        "--resume",
        str(run / "last.pt"),
    )
    assert (run / "code_snapshot" / "manifest.json").is_file()
    assert json.loads((run / "best_metrics.json").read_text())["study_count"] == 2


def test_full_data_fixed_epoch_and_resume(tmp_path: Path) -> None:
    _, run, config = build_experiment(tmp_path)
    settings = json.loads(config.read_text())
    settings.update(
        validation="none",
        merge_valid=True,
        backbone="efficientnet_b0",
        freeze_backbone=True,
        draft=False,
        schedule_epochs=12,
    )
    config.write_text(json.dumps(settings))
    invoke("train", "--config", str(config))
    manifest = json.loads((run / "split_manifest.json").read_text())
    assert len(manifest["train_study_ids"]) == 30
    assert manifest["valid_study_ids"] == []
    assert not (run / "best.pt").exists()
    state = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
    assert state["epoch"] == 1
    last = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    assert last["scheduler"]["T_max"] == 12
    assert state["best_metrics"] is None
    invoke("train", "--config", str(config), "--resume", str(run / "last.pt"))
    invoke(
        "predict",
        "--config",
        str(config),
        "--checkpoint",
        str(run / "final.pt"),
        "--split",
        "test",
        "--output",
        str(run / "submission.csv"),
    )
    assert (run / "submission.csv").is_file()
    root = Path(__file__).resolve().parents[2]
    package = tmp_path / "candidate"
    result = subprocess.run(
        [
            sys.executable,
            "tools/package_sprint_candidate.py",
            "--checkpoints",
            str(run / "final.pt"),
            "--data-root",
            str(tmp_path / "data"),
            "--output",
            str(package),
            "--device",
            "cpu",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len((package / "submission.csv").read_text().splitlines()) == 31
    assert (package / "training" / "0" / "code_snapshot" / "manifest.json").is_file()
    with zipfile.ZipFile(package.with_suffix(".zip")) as archive:
        assert "model/member_0.pt" in archive.namelist()
        assert "cost.py" in archive.namelist()


def test_experiment_feedback_archive(tmp_path: Path) -> None:
    _, run, config = build_experiment(tmp_path)
    settings = json.loads(config.read_text())
    settings.update(validation="holdout", backbone="resnet18", freeze_backbone=True)
    config.write_text(json.dumps(settings))
    holdout = write_holdout(tmp_path / "holdout.json", ["subject_train_000"], ["study_train_000"])
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "tools/run_sprint_experiment.py",
            "--config",
            str(config),
            "--holdout",
            str(holdout),
            "--allow-dirty",
        ],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((run / "experiment_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    assert receipt["source_unchanged"]
    assert receipt["test_inference"] is False
    with zipfile.ZipFile(run / "feedback.zip") as archive:
        assert "evaluation_dev_predictions.csv" in archive.namelist()
        assert "evaluation_valid_predictions.csv" in archive.namelist()
        assert "code_snapshot/manifest.json" in archive.namelist()
        assert not any(n.endswith(".pt") for n in archive.namelist())
