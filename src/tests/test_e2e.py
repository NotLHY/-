"""Summary: Exercise competition workflows and data rejection through the public CLI.

Description: Uses real JPEG fixtures and torchvision models without network weights.
Artifacts remain in pytest temporary directories for inspection after each run.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
FIELDS = [
    "Dicom_id",
    "Subject_id",
    "Study_id",
    "ViewPosition",
    "PatientOrientation",
    "Rows",
    "Columns",
    "Predict_class",
]


def invoke(*arguments: str, success: bool = True) -> subprocess.CompletedProcess[str]:
    """Run the real CLI in an isolated child process with bounded CPU use."""
    result = subprocess.run(
        [sys.executable, "-m", "cxr", *arguments],
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="2"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, "Invalid input was accepted: " + result.stdout
    return result


def build_experiment(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create disjoint patients, variable views, and all ten validation classes."""
    data_root = tmp_path / "data"
    for split, count in (("train", 20), ("valid", 10), ("test", 3)):
        rows = []
        for index in range(count):
            subject = f"subject_{split}_{index:03d}"
            study = f"study_{split}_{index:03d}"
            views = 1 + index % 3
            for view in range(views):
                image_id = f"pic_{split}_{index:03d}_{view}"
                path = data_root / split / subject / study / f"{image_id}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("L", (70 + view * 9, 90), 30 + (index * 11 + view * 5) % 200).save(path)
                row = dict(
                    zip(
                        FIELDS,
                        [
                            image_id,
                            subject,
                            study,
                            "AP" if view == 0 else "LATERAL",
                            "Erect",
                            90,
                            70 + view * 9,
                            str(index % 10 + 1),
                        ],
                        strict=True,
                    )
                )
                if split == "test":
                    del row["Predict_class"]
                rows.append(row)
        with (data_root / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS if split != "test" else FIELDS[:-1])
            writer.writeheader()
            writer.writerows(rows)
    output_dir = tmp_path / "run"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "name": "e2e_smoke",
                "data_root": str(data_root),
                "output_dir": str(output_dir),
                "pretrained": False,
                "pooling": "image_mean",
                "image_size": 64,
                "batch_size": 2,
                "epochs": 1,
                "device": "cpu",
                "amp": False,
                "num_workers": 0,
                "validation": "official",
                "max_train_batches": 1,
                "cpu_threads": 2,
            }
        ),
        encoding="utf-8",
    )
    return data_root, output_dir, config


@pytest.fixture
def experiment(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Expose the synthetic dataset to tests that request the fixture."""
    return build_experiment(tmp_path)


@pytest.mark.parametrize("pooling", ["image_mean", "study_mean", "study_attention"])
def test_train_reload_predict_and_resume(
    experiment: tuple[Path, Path, Path],
    pooling: str,
) -> None:
    """Training must produce usable weights, valid study scores, and resumable state."""
    _, output_dir, config = experiment
    settings = json.loads(config.read_text(encoding="utf-8"))
    settings["pooling"] = pooling
    config.write_text(json.dumps(settings), encoding="utf-8")
    invoke("train", "--config", str(config))
    for name in (
        "config.json",
        "environment.json",
        "split_manifest.json",
        "metrics.jsonl",
        "best.pt",
        "last.pt",
        "best_metrics.json",
        "validation_predictions.csv",
    ):
        assert (output_dir / name).is_file(), name
    metrics = json.loads((output_dir / "best_metrics.json").read_text(encoding="utf-8"))
    assert math.isfinite(metrics["macro_auc"])
    assert 0 <= metrics["macro_auc"] <= 1
    assert len(metrics["per_class_auc"]) == 10
    invoke("evaluate", "--config", str(config), "--checkpoint", str(output_dir / "best.pt"))
    saved_rows = list(
        csv.DictReader((output_dir / "validation_predictions.csv").open(encoding="utf-8"))
    )
    reloaded_rows = list(
        csv.DictReader((output_dir / "evaluation_valid_predictions.csv").open(encoding="utf-8"))
    )
    saved_scores = [float(row["Probability"]) for row in saved_rows]
    assert [float(row["Probability"]) for row in reloaded_rows] == pytest.approx(
        saved_scores, abs=2e-5
    )
    by_study: dict[str, list[float]] = {}
    for row in saved_rows:
        by_study.setdefault(row["Study_id"], []).append(float(row["Probability"]))
    truth = [[int(label == index % 10) for label in range(10)] for index in range(10)]
    independent_auc = roc_auc_score(truth, list(by_study.values()), average="macro")
    assert metrics["macro_auc"] == pytest.approx(independent_auc)
    destination = output_dir / "smoke_submission.csv"
    invoke(
        "predict",
        "--config",
        str(config),
        "--checkpoint",
        str(output_dir / "best.pt"),
        "--split",
        "test",
        "--output",
        str(destination),
    )
    rows = list(csv.DictReader(destination.open(encoding="utf-8")))
    assert len(rows) == 30
    assert list(rows[0]) == ["Study_id", "Predict_class", "Probability"]
    assert len({(row["Study_id"], row["Predict_class"]) for row in rows}) == 30
    assert all(
        math.isfinite(float(row["Probability"])) and 0 <= float(row["Probability"]) <= 1
        for row in rows
    )
    settings["batch_size"] = 1
    config.write_text(json.dumps(settings), encoding="utf-8")
    single = output_dir / "single_batch.csv"
    invoke(
        "predict",
        "--config",
        str(config),
        "--checkpoint",
        str(output_dir / "best.pt"),
        "--split",
        "test",
        "--output",
        str(single),
        "--include-subject",
    )
    one_rows = list(csv.DictReader(single.open(encoding="utf-8")))
    assert "Subject_id" in one_rows[0]
    assert [float(row["Probability"]) for row in one_rows] == pytest.approx(
        [float(row["Probability"]) for row in rows],
        abs=2e-5,
    )
    settings["epochs"] = 2
    settings["batch_size"] = 2
    config.write_text(json.dumps(settings), encoding="utf-8")
    invoke("train", "--config", str(config), "--resume", str(output_dir / "last.pt"))
    history = [json.loads(line) for line in (output_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(history) == 2
    assert history[-1]["epoch"] == 2
    assert (output_dir / "last.pt").stat().st_size > 0


@pytest.mark.parametrize(
    "failure", ["normal_overlap", "missing_file", "path_traversal", "patient_overlap"]
)
def test_audit_rejects_invalid_data(experiment: tuple[Path, Path, Path], failure: str) -> None:
    """Bad labels, paths, files, and patient leakage must fail before training."""
    data_root, _, config = experiment
    csv_path = data_root / "train.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if failure == "normal_overlap":
        rows[0]["Predict_class"] = "6;10"
    elif failure == "missing_file":
        rows[0]["Dicom_id"] = "pic_missing"
    elif failure == "path_traversal":
        rows[0]["Subject_id"] = "../escape"
    else:
        valid_csv = data_root / "valid.csv"
        valid_rows = list(csv.DictReader(valid_csv.open(encoding="utf-8")))
        old = data_root / "valid" / valid_rows[0]["Subject_id"]
        new = data_root / "valid" / rows[0]["Subject_id"]
        old.rename(new)
        valid_rows[0]["Subject_id"] = rows[0]["Subject_id"]
        with valid_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(valid_rows)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    invoke("audit", "--config", str(config), success=False)


def test_class_one_can_coexist_with_disease(experiment: tuple[Path, Path, Path]) -> None:
    """Class 1 is enlarged cardiomediastinum, whereas class 10 is No Finding."""
    data_root, output_dir, config = experiment
    csv_path = data_root / "train.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    rows[0]["Predict_class"] = "1;6"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    invoke("audit", "--config", str(config))
    report = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert report["splits"]["train"]["studies"] == 20
    assert report["splits"]["train"]["label_positives"][5] == 3


def test_unknown_config_key_is_rejected(experiment: tuple[Path, Path, Path]) -> None:
    """Misspelled experiment knobs cannot silently change the intended experiment."""
    _, _, config = experiment
    settings = json.loads(config.read_text(encoding="utf-8"))
    settings["batch_szie"] = 4
    config.write_text(json.dumps(settings), encoding="utf-8")
    invoke("audit", "--config", str(config), success=False)
