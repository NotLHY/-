"""Ten-class validation metrics with optional test-view reweighting."""

from __future__ import annotations

from collections.abc import Sequence

try:  # Python 3.11+
    from typing import NotRequired, TypedDict
except ImportError:  # 3.10 local conda envs; typing_extensions backports both
    from typing_extensions import NotRequired, TypedDict

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


class Metrics(TypedDict):
    val_loss: float
    macro_auc: float | None
    macro_auc_defined: float | None
    defined_auc_count: int
    macro_auc_reweighted: float | None
    macro_auc_reweighted_defined: float | None
    defined_auc_count_reweighted: int
    macro_f1: float
    per_class_auc: list[float | None]
    per_class_auc_reweighted: list[float | None]
    view_group_auc: dict[str, float | None]
    view_weights: dict[str, float] | None
    positive_counts: list[int]
    study_count: int
    epoch: NotRequired[int]
    train_loss: NotRequired[float]
    steps: NotRequired[int]
    elapsed_seconds: NotRequired[float]
    partial_smoke: NotRequired[bool]
    split: NotRequired[str]
    checkpoint: NotRequired[str]
    scope: NotRequired[str]
    selection_metric: NotRequired[str]


def _auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    mask = np.ones(len(labels), dtype=bool) if weights is None else weights > 0
    if len(np.unique(labels[mask])) < 2 or len(labels[mask]) < 2:
        return None
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def summarize(
    targets: np.ndarray,
    probabilities: np.ndarray,
    val_loss: float,
    view_groups: Sequence[str] | None = None,
    view_weights: dict[str, float] | None = None,
) -> Metrics:
    if targets.ndim != 2 or targets.shape[1] != 10 or probabilities.shape != targets.shape:
        raise ValueError("Expected matching N x 10 targets and probabilities")
    if not np.isfinite(probabilities).all() or not np.isfinite(val_loss):
        raise ValueError("Nonfinite probabilities or validation loss")
    if view_groups is not None and len(view_groups) != targets.shape[0]:
        raise ValueError("view_groups must match the number of studies")

    aucs: list[float | None] = []
    for index in range(10):
        aucs.append(_auc(targets[:, index], probabilities[:, index]))
    defined = [value for value in aucs if value is not None]

    weights: np.ndarray | None = None
    reweighted_aucs: list[float | None] = []
    group_auc: dict[str, float | None] = {}
    if view_groups is not None and view_weights is not None:
        weights = np.asarray([view_weights.get(group, 0.0) for group in view_groups])
        for index in range(10):
            reweighted_aucs.append(
                _auc(targets[:, index], probabilities[:, index], weights)
            )
        for group in sorted(set(view_groups)):
            selected = np.asarray([value == group for value in view_groups])
            if selected.sum() < 2:
                group_auc[group] = None
                continue
            per_class = [
                _auc(targets[selected, index], probabilities[selected, index])
                for index in range(10)
            ]
            defined_group = [value for value in per_class if value is not None]
            group_auc[group] = float(np.mean(defined_group)) if defined_group else None
    else:
        reweighted_aucs = [None] * 10

    defined_reweighted = [value for value in reweighted_aucs if value is not None]
    return {
        "val_loss": float(val_loss),
        "macro_auc": float(np.mean(defined)) if len(defined) == 10 else None,
        "macro_auc_defined": float(np.mean(defined)) if defined else None,
        "defined_auc_count": len(defined),
        "macro_auc_reweighted": (
            float(np.mean(defined_reweighted)) if len(defined_reweighted) == 10 else None
        ),
        "macro_auc_reweighted_defined": (
            float(np.mean(defined_reweighted)) if defined_reweighted else None
        ),
        "defined_auc_count_reweighted": len(defined_reweighted),
        "macro_f1": float(
            f1_score(targets, probabilities >= 0.5, average="macro", zero_division=0)
        ),
        "per_class_auc": aucs,
        "per_class_auc_reweighted": reweighted_aucs,
        "view_group_auc": group_auc,
        "view_weights": view_weights,
        "positive_counts": targets.sum(axis=0).astype(int).tolist(),
        "study_count": int(targets.shape[0]),
    }
