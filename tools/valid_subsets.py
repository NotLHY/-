"""Report official-valid Macro-AUC overall and on frequent-patient subsets.

The hidden test cohort is dominated by patients with many repeat checks.  Overall
valid AUC is therefore optimistic for test-like patients.  This tool reports the
same fixed model on all patients and on >=5/>=8/>=10 checks-per-patient subsets,
with a patient-cluster bootstrap.  It reads only valid.csv and a run's
``evaluation_valid_predictions.csv``; test images, labels and test-derived weights
are never used.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cxr.metrics import summarize  # noqa: E402

CLASSES = tuple(range(1, 11))


def load_truth(path: Path) -> tuple[dict[str, str], dict[str, set[int]]]:
    subject: dict[str, str] = {}
    labels: dict[str, set[int]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            study = row["Study_id"]
            item_labels = {int(token) for token in row["Predict_class"].split(";") if token}
            previous = labels.setdefault(study, item_labels)
            if previous != item_labels:
                raise ValueError(f"Inconsistent labels within study {study}")
            subject[study] = row["Subject_id"]
    if not labels:
        raise ValueError(f"No labeled studies in {path}")
    return subject, labels


def load_predictions(path: Path) -> dict[str, dict[int, float]]:
    predictions: dict[str, dict[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            study = row["Study_id"]
            label = int(row["Predict_class"])
            probability = float(row["Probability"])
            if label not in CLASSES:
                raise ValueError(f"{path}: class outside 1..10 for {study}")
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"{path}: probability outside [0,1] for {study}/{label}")
            scores = predictions.setdefault(study, {})
            if label in scores:
                raise ValueError(f"{path}: duplicate probability for {study}/{label}")
            scores[label] = probability
    return predictions


def matrix(
    studies: list[str],
    labels: dict[str, set[int]],
    predictions: dict[str, dict[int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.zeros((len(studies), len(CLASSES)), dtype=np.int8)
    scores = np.zeros((len(studies), len(CLASSES)), dtype=np.float64)
    for index, study in enumerate(studies):
        for column, label in enumerate(CLASSES):
            truth[index, column] = int(label in labels[study])
            scores[index, column] = predictions[study][label]
    return truth, scores


def metric_row(truth: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    metrics = summarize(truth, scores, 0.0)
    return {
        "studies": int(truth.shape[0]),
        "macro_auc": metrics["macro_auc"],
        "macro_auc_defined": metrics["macro_auc_defined"],
        "defined_classes": metrics["defined_auc_count"],
        "per_class_auc": metrics["per_class_auc"],
    }


def bootstrap_ci(
    studies: list[str],
    subject: dict[str, str],
    labels: dict[str, set[int]],
    predictions: dict[str, dict[int, float]],
    keep: set[str],
    reps: int,
    seed: int,
) -> tuple[list[float] | None, int]:
    groups: dict[str, list[int]] = {}
    for index, study in enumerate(studies):
        if subject[study] in keep:
            groups.setdefault(subject[study], []).append(index)
    names = sorted(groups)
    if not names:
        return None, 0
    truth, scores = matrix(studies, labels, predictions)
    rng = np.random.RandomState(seed)
    values: list[float] = []
    skipped = 0
    for _ in range(reps):
        picked = rng.choice(len(names), size=len(names), replace=True)
        indices = np.concatenate([groups[names[index]] for index in picked])
        metrics = summarize(truth[indices], scores[indices], 0.0)
        macro = metrics["macro_auc"]
        if macro is None:
            skipped += 1
            continue
        values.append(float(macro))
    if not values:
        return None, skipped
    low, high = np.percentile(np.asarray(values), [2.5, 97.5])
    return [float(low), float(high)], skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid", type=Path, required=True, help="official valid.csv")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")

    subject, labels = load_truth(args.valid)
    predictions = load_predictions(args.predictions)
    studies = sorted(labels)
    if set(predictions) != set(studies):
        raise ValueError("Prediction study ids do not match valid.csv")
    missing = [study for study in studies if len(predictions[study]) != 10]
    if missing:
        raise ValueError(f"{len(missing)} studies are missing class probabilities")

    counts = Counter(subject[study] for study in studies)
    subsets = {
        "overall": set(counts),
        "ge5_checks_per_patient": {name for name, count in counts.items() if count >= 5},
        "ge8_checks_per_patient": {name for name, count in counts.items() if count >= 8},
        "ge10_checks_per_patient": {name for name, count in counts.items() if count >= 10},
    }

    report: dict[str, object] = {
        "valid_studies": len(studies),
        "valid_patients": len(counts),
        "note": "Subsets use only valid.csv patient/check counts; no test labels or test-derived weights.",
        "subsets": {},
    }
    truth_all, scores_all = matrix(studies, labels, predictions)
    print(f"valid studies={len(studies)} patients={len(counts)}")
    print(f"{'subset':<26} {'studies':>7} {'patients':>8} {'macro_auc':>10} {'95% CI':>20}")
    for name, keep in subsets.items():
        chosen = [index for index, study in enumerate(studies) if subject[study] in keep]
        truth = truth_all[chosen]
        scores = scores_all[chosen]
        row = metric_row(truth, scores)
        ci, skipped = bootstrap_ci(
            studies, subject, labels, predictions, keep, args.reps, args.seed
        )
        row["patients"] = len(keep)
        row["bootstrap_ci95"] = ci
        row["bootstrap_reps"] = args.reps
        row["bootstrap_skipped"] = skipped
        report["subsets"][name] = row  # type: ignore[index]
        macro = row["macro_auc"]
        ci_text = "None" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        macro_text = "None" if macro is None else f"{float(macro):.4f}"
        print(
            f"{name:<26} {row['studies']:>7} {row['patients']:>8} "
            f"{macro_text:>10} {ci_text:>20}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
