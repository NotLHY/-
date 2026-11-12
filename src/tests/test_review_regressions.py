"""Verify saved evaluation scope and interrupted-run recovery through the CLI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from test_e2e import build_experiment, invoke


def test_saved_scope_and_training_provenance(tmp_path: Path) -> None:
    """Default evaluation reproduces saved scope and retains checkpoint smoke provenance."""
    _, run, config = build_experiment(tmp_path)
    settings = json.loads(config.read_text(encoding="utf-8"))
    settings.update(backbone="resnet18", freeze_backbone=True, max_val_batches=1)
    config.write_text(json.dumps(settings), encoding="utf-8")
    invoke("train", "--config", str(config))
    saved = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
    assert saved["study_count"] == 2
    invoke("evaluate", "--config", str(config), "--checkpoint", str(run / "best.pt"))
    evaluated = json.loads((run / "evaluation_valid.json").read_text(encoding="utf-8"))
    assert evaluated["study_count"] == saved["study_count"]
    settings.update(pretrained=True, max_train_batches=None, max_val_batches=None)
    config.write_text(json.dumps(settings), encoding="utf-8")
    invoke("evaluate", "--config", str(config), "--checkpoint", str(run / "best.pt"),
           "--split", "valid")
    evaluated = json.loads((run / "evaluation_valid.json").read_text(encoding="utf-8"))
    assert evaluated["study_count"] == 10
    assert evaluated["partial_smoke"] is True


def test_resume_reconciles_uncommitted_epoch_log(tmp_path: Path) -> None:
    """A log ahead of last.pt must not produce duplicate epochs after recovery."""
    _, run, config = build_experiment(tmp_path)
    settings = json.loads(config.read_text(encoding="utf-8"))
    settings.update(backbone="resnet18", freeze_backbone=True)
    config.write_text(json.dumps(settings), encoding="utf-8")
    invoke("train", "--config", str(config))
    committed = run / "committed_epoch1.pt"
    shutil.copyfile(run / "last.pt", committed)
    settings["epochs"] = 2
    config.write_text(json.dumps(settings), encoding="utf-8")
    invoke("train", "--config", str(config), "--resume", str(run / "last.pt"))
    shutil.copyfile(committed, run / "last.pt")
    invoke("train", "--config", str(config), "--resume", str(run / "last.pt"))
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2]
