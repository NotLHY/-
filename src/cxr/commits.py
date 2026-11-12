"""Reconcile run artifacts to the epoch committed by last.pt."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import torch

from .artifacts import write_checkpoint, write_json, write_predictions
from .metrics import Metrics
from .records import StudyRecord


def snapshot_model(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _write_rows(path: Path, rows: list[Metrics]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def reconcile_log(path: Path, state: dict[str, Any]) -> None:
    """Keep only unique committed rows; checkpoint stores the latest committed row."""
    epoch = int(state["epoch"])
    rows: dict[int, Metrics] = {}
    previous = 0
    if path.exists():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                if len(rows) >= epoch:
                    break
                raise ValueError(f"Corrupt committed metrics.jsonl line {number}: blank row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                if len(rows) >= epoch:
                    break
                raise ValueError(f"Corrupt committed metrics.jsonl line {number}") from error
            if not isinstance(row, dict) or type(row.get("epoch")) is not int:
                raise ValueError(f"Corrupt metrics.jsonl line {number}: invalid epoch")
            row_epoch = row["epoch"]
            if row_epoch > epoch:
                break
            if row_epoch < 1 or row_epoch < previous:
                raise ValueError(f"Impossible committed metrics.jsonl order at line {number}")
            if row_epoch in rows and rows[row_epoch] != row:
                raise ValueError(f"Conflicting committed metrics.jsonl epoch {row_epoch}")
            rows[row_epoch] = cast(Metrics, row)
            previous = row_epoch
    current = state.get("epoch_metrics")
    if epoch and current is not None:
        if current.get("epoch") != epoch:
            raise ValueError("Checkpoint epoch_metrics differs from committed epoch")
        if epoch in rows and rows[epoch] != current:
            raise ValueError(f"Committed metrics.jsonl epoch {epoch} differs from last.pt")
        rows[epoch] = current
    missing = [index for index in range(1, epoch + 1) if index not in rows]
    if missing:
        raise ValueError(f"Missing committed metrics.jsonl epochs {missing}; cannot reconstruct")
    ordered = [rows[index] for index in range(1, epoch + 1)]
    expected = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in ordered
    )
    if not path.exists() or path.read_text(encoding="utf-8") != expected:
        _write_rows(path, ordered)


def reconcile_best(run: Path, state: dict[str, Any], studies: list[StudyRecord]) -> None:
    """The last checkpoint owns the best model and probabilities for recovery."""
    paths = [run / name for name in ("best.pt", "best_metrics.json", "validation_predictions.csv")]
    metrics: Metrics | None = state["best_metrics"]
    if metrics is None:
        for path in paths:
            path.unlink(missing_ok=True)
        return
    best_epoch = int(metrics["epoch"])
    if best_epoch < 1 or best_epoch > int(state["epoch"]):
        raise ValueError("Checkpoint best epoch exceeds committed epoch")
    ids: list[str] | None = state.get("best_validation_study_ids")
    probabilities: torch.Tensor | None = state.get("best_probabilities")
    best_model: dict[str, torch.Tensor] | None = state.get("best_model")
    if ids is None or probabilities is None or best_model is None:
        raise ValueError("Checkpoint lacks best model, validation scope, or probabilities")
    selected = {study.study_id: study for study in studies}
    if len(ids) != len(set(ids)) or any(key not in selected for key in ids):
        raise ValueError("Checkpoint best validation scope is invalid for saved manifest")
    scoped = [selected[key] for key in ids]
    if len(scoped) != metrics["study_count"]:
        raise ValueError("Checkpoint best scope differs from best metrics")
    best_path = paths[0]
    existing_epoch = None
    if best_path.exists():
        best_state = torch.load(best_path, map_location="cpu", weights_only=True)
        existing_epoch = best_state.get("epoch")
    if existing_epoch != best_epoch:
        write_checkpoint(best_path, {
            "model": best_model,
            "epoch": best_epoch,
            "best_metrics": metrics,
            "config": state["config"],
            "validation_study_ids": ids,
        })
    write_json(paths[1], metrics)
    write_predictions(paths[2], scoped, probabilities.numpy())


def reconcile_run(run: Path, state: dict[str, Any], studies: list[StudyRecord]) -> None:
    reconcile_log(run / "metrics.jsonl", state)
    reconcile_best(run, state, studies)
