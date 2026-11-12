"""Held-out longitudinal split: exclusion from training and scoring scope at evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_e2e import build_experiment, invoke


def write_holdout(path: Path, subjects: list[str], studies: list[str]) -> Path:
    path.write_text(
        json.dumps({"version": 1, "subjects": subjects, "studies": studies}),
        encoding="utf-8",
    )
    return path


def test_train_excludes_holdout_patients(tmp_path: Path) -> None:
    """Held-out patients must never reach the training set, and the manifest says so."""
    data, run, config = build_experiment(tmp_path)
    settings = json.loads(config.read_text())
    settings.update(backbone="resnet18", freeze_backbone=True)
    config.write_text(json.dumps(settings))
    holdout = write_holdout(
        tmp_path / "holdout.json",
        ["subject_train_000", "subject_train_001"],
        ["study_train_000", "study_train_001"],
    )
    invoke("train", "--config", str(config), "--holdout-manifest", str(holdout))
    manifest = json.loads((run / "split_manifest.json").read_text())
    train_subjects = set(manifest["train_subject_ids"])
    assert "subject_train_000" not in train_subjects
    assert "subject_train_001" not in train_subjects
    assert len(train_subjects) == 18
    assert manifest["holdout"]["subjects"] == 2
    assert manifest["holdout"]["excluded_studies"] == 2


def test_evaluate_scores_only_holdout_studies(tmp_path: Path) -> None:
    """Evaluation with a holdout manifest reports the dev scope, not the whole split."""
    _, run, config = build_experiment(tmp_path)
    settings = json.loads(config.read_text())
    settings.update(backbone="resnet18", freeze_backbone=True)
    config.write_text(json.dumps(settings))
    holdout = write_holdout(
        tmp_path / "holdout.json",
        ["subject_train_002", "subject_train_003", "subject_train_004"],
        ["study_train_002", "study_train_003", "study_train_004"],
    )
    invoke("train", "--config", str(config), "--holdout-manifest", str(holdout))
    invoke(
        "evaluate",
        "--config",
        str(config),
        "--checkpoint",
        str(run / "best.pt"),
        "--split",
        "train",
        "--holdout-manifest",
        str(holdout),
    )
    metrics = json.loads((run / "evaluation_dev.json").read_text())
    assert metrics["split"] == "dev"
    assert metrics["study_count"] == 3
    predictions = (run / "evaluation_dev_predictions.csv").read_text().splitlines()
    assert len(predictions) == 3 * 10 + 1


def test_holdout_rejects_unknown_study(tmp_path: Path) -> None:
    """A manifest listing a study that the split does not contain must fail loudly."""
    _, run, config = build_experiment(tmp_path)
    settings = json.loads(config.read_text())
    settings.update(backbone="resnet18", freeze_backbone=True)
    config.write_text(json.dumps(settings))
    holdout = write_holdout(
        tmp_path / "holdout.json",
        ["subject_train_005"],
        ["study_train_005", "study_that_does_not_exist"],
    )
    invoke("train", "--config", str(config), "--holdout-manifest", str(holdout))
    result = invoke(
        "evaluate",
        "--config",
        str(config),
        "--checkpoint",
        str(run / "best.pt"),
        "--split",
        "train",
        "--holdout-manifest",
        str(holdout),
        success=False,
    )
    assert "absent from the split" in result.stderr


def test_holdout_rejects_patient_in_validation(tmp_path: Path) -> None:
    """A patient that already sits in valid cannot also be held out of train."""
    data, run, config = build_experiment(tmp_path)
    settings = json.loads(config.read_text())
    settings.update(backbone="resnet18", freeze_backbone=True)
    config.write_text(json.dumps(settings))
    holdout = write_holdout(
        tmp_path / "holdout.json",
        ["subject_valid_000"],
        ["study_valid_000"],
    )
    result = invoke(
        "train", "--config", str(config), "--holdout-manifest", str(holdout), success=False
    )
    assert "appear in" in result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
