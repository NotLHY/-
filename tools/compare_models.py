"""Compare two models on the official valid split: per-class AUC, patient-cluster
paired bootstrap, and a >=3-checks-per-patient stress test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.metrics import summarize  # noqa: E402

CLASSES = list(range(1, 11))


def load_truth(path: Path) -> tuple[dict[str, str], dict[str, set[int]]]:
    subject: dict[str, str] = {}
    labels: dict[str, set[int]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            study = row["Study_id"]
            subject[study] = row["Subject_id"]
            labels[study] = {int(token) for token in row["Predict_class"].split(";") if token}
    return subject, labels


def load_predictions(path: Path) -> dict[str, dict[int, float]]:
    predictions: dict[str, dict[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            study, label = row["Study_id"], int(row["Predict_class"])
            probability = float(row["Probability"])
            if label not in CLASSES:
                raise ValueError(f"Invalid classes for {study}: {label}")
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Probability outside finite [0,1] for {study}/{label}")
            scores = predictions.setdefault(study, {})
            if label in scores:
                raise ValueError(f"Duplicate probability for {study}/{label}")
            scores[label] = probability
    return predictions


def matrices(
    studies: list[str], labels: dict[str, set[int]], predictions: dict[str, dict[int, float]]
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.zeros((len(studies), len(CLASSES)), dtype=np.int8)
    scores = np.zeros((len(studies), len(CLASSES)), dtype=np.float64)
    for index, study in enumerate(studies):
        for column, label in enumerate(CLASSES):
            truth[index, column] = int(label in labels[study])
            scores[index, column] = predictions[study][label]
    return truth, scores


def class_aucs(truth: np.ndarray, scores: np.ndarray) -> list[float]:
    if not len(truth):
        return [float("nan")] * 10
    metrics = summarize(truth, scores, 0.0)
    return [value if value is not None else float("nan") for value in metrics["per_class_auc"]]


def macro(values: list[float]) -> tuple[float, int]:
    defined = [value for value in values if not math.isnan(value)]
    return (float(np.mean(defined)) if len(defined) == 10 else float("nan"), len(defined))


def nullable(value: float) -> float | None:
    """Undefined AUC is represented as null in strict JSON, never a partial-class mean."""
    return None if math.isnan(value) else value


def subset_metrics(
    studies: list[str],
    subject: dict[str, str],
    labels: dict[str, set[int]],
    base: dict[str, dict[int, float]],
    asl: dict[str, dict[int, float]],
    keep: set[str],
) -> dict[str, object]:
    chosen = [index for index, study in enumerate(studies) if subject[study] in keep]
    truth, base_scores = matrices([studies[i] for i in chosen], labels, base)
    _, asl_scores = matrices([studies[i] for i in chosen], labels, asl)
    base_aucs = class_aucs(truth, base_scores)
    asl_aucs = class_aucs(truth, asl_scores)
    base_macro, base_defined = macro(base_aucs)
    asl_macro, asl_defined = macro(asl_aucs)
    return {
        "studies": len(chosen),
        "patients": len(keep),
        "baseline": {
            "per_class": [nullable(x) for x in base_aucs],
            "macro": nullable(base_macro),
            "defined": base_defined,
        },
        "asl": {
            "per_class": [nullable(x) for x in asl_aucs],
            "macro": nullable(asl_macro),
            "defined": asl_defined,
        },
        "delta": nullable(asl_macro - base_macro),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--asl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")

    subject, labels = load_truth(args.valid)
    base = load_predictions(args.baseline)
    asl = load_predictions(args.asl)
    studies = sorted(labels)
    if not studies:
        raise ValueError("No labeled studies")
    if set(base) != set(studies) or set(asl) != set(studies):
        raise ValueError("Prediction study ids do not match valid.csv")
    missing = [s for s in studies if len(base[s]) != 10 or len(asl[s]) != 10]
    if missing:
        raise ValueError(f"Predictions missing classes for {len(missing)} studies")

    truth, base_scores = matrices(studies, labels, base)
    _, asl_scores = matrices(studies, labels, asl)
    base_aucs = class_aucs(truth, base_scores)
    asl_aucs = class_aucs(truth, asl_scores)
    base_macro, base_defined = macro(base_aucs)
    asl_macro, asl_defined = macro(asl_aucs)

    groups: dict[str, list[int]] = {}
    for index, study in enumerate(studies):
        groups.setdefault(subject[study], []).append(index)
    subjects = sorted(groups)
    rng = np.random.RandomState(args.seed)
    diffs: list[float] = []
    base_reps: list[float] = []
    asl_reps: list[float] = []
    skipped = 0
    for _ in range(args.reps):
        picked = rng.choice(len(subjects), size=len(subjects), replace=True)
        indices = np.concatenate([groups[subjects[i]] for i in picked])
        resample_truth = truth[indices]
        positives = resample_truth.sum(axis=0)
        if np.any(positives == 0) or np.any(positives == len(resample_truth)):
            skipped += 1
            continue
        first = class_aucs(resample_truth, base_scores[indices])
        second = class_aucs(resample_truth, asl_scores[indices])
        first_macro, _ = macro(first)
        second_macro, _ = macro(second)
        if math.isnan(first_macro) or math.isnan(second_macro):
            raise RuntimeError("Complete resample unexpectedly has undefined AUC")
        base_reps.append(first_macro)
        asl_reps.append(second_macro)
        diffs.append(second_macro - first_macro)
    diffs_array = np.asarray(diffs)
    low, high = np.percentile(diffs_array, [2.5, 97.5]) if len(diffs) else (float("nan"),) * 2
    fraction_nonpositive = float((diffs_array <= 0).mean()) if len(diffs) else None

    counts = Counter(subject[study] for study in studies)
    frequent = {name for name, count in counts.items() if count >= 3}

    result = {
        "valid_studies": len(studies),
        "valid_patients": len(subjects),
        "baseline": {
            "per_class": [nullable(x) for x in base_aucs],
            "macro": nullable(base_macro),
            "defined": base_defined,
        },
        "asl": {
            "per_class": [nullable(x) for x in asl_aucs],
            "macro": nullable(asl_macro),
            "defined": asl_defined,
        },
        "delta_macro": nullable(asl_macro - base_macro),
        "bootstrap": {
            "reps_requested": args.reps,
            "reps_used": len(diffs),
            "reps_skipped_missing_class": skipped,
            "mean_diff": float(diffs_array.mean()) if len(diffs) else None,
            "ci95": [float(low), float(high)] if len(diffs) else None,
            "fraction_delta_le_zero": fraction_nonpositive,
            "baseline_mean": float(np.mean(base_reps)) if base_reps else None,
            "asl_mean": float(np.mean(asl_reps)) if asl_reps else None,
            "note": "Patient-paired percentile interval conditional on all ten classes; "
            "nonpositive fraction is descriptive, not a calibrated p-value.",
        },
        "stress_ge3_checks_per_patient": subset_metrics(
            studies, subject, labels, base, asl, frequent
        ),
        "per_class_delta": [
            (asl_aucs[i] - base_aucs[i]) if not math.isnan(asl_aucs[i] - base_aucs[i]) else None
            for i in range(len(CLASSES))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    print(f"studies={result['valid_studies']} patients={result['valid_patients']}")
    print(f"baseline macro={base_macro:.4f} ({base_defined}/10)")
    print(f"ASL      macro={asl_macro:.4f} ({asl_defined}/10)")
    print(f"delta={result['delta_macro']}")
    print(
        "patient-cluster bootstrap: "
        f"mean={result['bootstrap']['mean_diff']} "
        f"95%CI={result['bootstrap']['ci95']} "
        f"used={len(diffs)} skipped_missing_class={skipped}"
    )
    stress = result["stress_ge3_checks_per_patient"]
    print(
        f"stress >=3 checks/patient: studies={stress['studies']} patients={stress['patients']} "
        f"baseline={stress['baseline']['macro']} asl={stress['asl']['macro']} "
        f"delta={stress['delta']}"
    )
    print("class  baseline    asl    delta")
    for index, label in enumerate(CLASSES):
        print(
            f"{label:>5}  {base_aucs[index]:.4f}  {asl_aucs[index]:.4f}  "
            f"{asl_aucs[index] - base_aucs[index]:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
