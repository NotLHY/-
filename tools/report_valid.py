"""Sweep every cached validation prediction: per-class error profile and ensemble ceiling.

Reads only `runs/*/evaluation_valid_predictions.csv` plus `valid.csv`. No model, no GPU.

What it answers
---------------
1. which of the 28 runs is strongest on the official valid split, overall and on the
   frequent-visitor subsets that look like the hidden test cohort;
2. how much an equal-weight rank ensemble of the existing checkpoints buys, and which
   members it picks (greedy forward selection, so the number is an **optimistic**
   in-sample estimate -- it is selected on the same split it is scored on);
3. which classes drag the macro average, i.e. where the next experiment should go;
4. a patient-cluster paired bootstrap of the ensemble against the best single model.

Example
-------
    python tools/report_valid.py \
        --valid cxr_first/cxr_first/valid.csv \
        --runs runs --output runs/_report/valid_sweep.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.metrics import summarize  # noqa: E402

CLASSES = list(range(1, 11))
RNG = np.random.default_rng(20261103)


def load_truth(path: Path) -> tuple[dict[str, str], dict[str, set[int]]]:
    subject: dict[str, str] = {}
    labels: dict[str, set[int]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            study = row["Study_id"]
            subject[study] = row["Subject_id"]
            for token in row["Predict_class"].split(";"):
                token = token.strip()
                if token:
                    labels[study].add(int(token))
    return subject, labels


def load_predictions(path: Path) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            out.setdefault(row["Study_id"], {})[int(row["Predict_class"])] = float(
                row["Probability"]
            )
    return out


def to_matrix(studies: list[str], predictions: dict[str, dict[int, float]]) -> np.ndarray:
    matrix = np.full((len(studies), 10), np.nan)
    for row, study in enumerate(studies):
        for column, label in enumerate(CLASSES):
            matrix[row, column] = predictions.get(study, {}).get(label, np.nan)
    if np.isnan(matrix).any():
        missing = int(np.isnan(matrix).any(axis=1).sum())
        raise ValueError(f"{missing} studies missing scores")
    return matrix


def rank_normalise(matrix: np.ndarray) -> np.ndarray:
    out = np.empty_like(matrix)
    for column in range(matrix.shape[1]):
        order = np.argsort(np.argsort(matrix[:, column]))
        out[:, column] = order / max(len(order) - 1, 1)
    return out


def macro(targets: np.ndarray, scores: np.ndarray) -> float | None:
    return summarize(targets, scores, 0.0)["macro_auc"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--max-members", type=int, default=6)
    parser.add_argument("--reps", type=int, default=400)
    args = parser.parse_args()

    subject, labels = load_truth(args.valid)
    paths = sorted(args.runs.glob("*/evaluation_valid_predictions.csv"))
    skipped_partial = []
    kept = []
    for path in paths:
        report = path.parent / "evaluation_valid.json"
        if report.is_file():
            try:
                if json.loads(report.read_text(encoding="utf-8")).get("partial_smoke"):
                    skipped_partial.append(path.parent.name)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        kept.append(path)
    paths = kept
    if skipped_partial:
        print(f"skipped partial smoke runs: {', '.join(sorted(skipped_partial))}")

    # one common study order, taken from the first file that has all 859 studies
    runs: dict[str, dict[str, dict[int, float]]] = {}
    for path in paths:
        runs[path.parent.name] = load_predictions(path)
    if not runs:
        raise SystemExit(f"no evaluation_valid_predictions.csv under {args.runs}")
    studies = sorted({study for table in runs.values() for study in table})
    studies = [s for s in studies if s in subject]
    studies.sort()
    print(f"runs={len(runs)}  studies={len(studies)}  patients={len({subject[s] for s in studies})}")

    targets = np.array([[1.0 if label in labels[s] else 0.0 for label in CLASSES] for s in studies])
    patients = np.array([subject[s] for s in studies])
    per_patient = defaultdict(int)
    for p in patients:
        per_patient[p] += 1
    counts = np.array([per_patient[p] for p in patients])

    scores: dict[str, np.ndarray] = {}
    rows = []
    for name, table in runs.items():
        try:
            matrix = to_matrix(studies, table)
        except ValueError:
            continue
        scores[name] = matrix
        metrics = summarize(targets, matrix, 0.0)
        entry = {
            "run": name,
            "macro": metrics["macro_auc"],
            "per_class": metrics["per_class_auc"],
            "positive_counts": metrics["positive_counts"],
        }
        for key, mask in (("ge5", counts >= 5), ("ge10", counts >= 10)):
            entry[key] = macro(targets[mask], matrix[mask])
        rows.append(entry)

    rows.sort(key=lambda r: (r["macro"] or 0), reverse=True)
    print("\n== single runs on official valid (sorted by macro) ==")
    print(f"{'run':<28}{'macro':>7}{'>=5':>7}{'>=10':>7}")
    for row in rows[: args.top]:
        print(
            f"{row['run']:<28}{row['macro'] or 0:7.4f}"
            f"{(row['ge5'] or 0):7.4f}{(row['ge10'] or 0):7.4f}"
        )

    ranks = {name: rank_normalise(matrix) for name, matrix in scores.items()}
    current = np.zeros_like(targets)
    chosen: list[str] = []
    best_score = 0.0
    curve = []
    for _ in range(args.max_members):
        best_name, best_value, best_matrix = None, best_score, None
        for name, matrix in ranks.items():
            if name in chosen:
                continue
            trial = (current * len(chosen) + matrix) / (len(chosen) + 1)
            value = macro(targets, trial) or 0.0
            if value > best_value:
                best_name, best_value, best_matrix = name, value, trial
        if best_name is None:
            break
        chosen.append(best_name)
        current = best_matrix if best_matrix is not None else current
        best_score = best_value
        curve.append({"members": list(chosen), "macro": best_value})
        print(f"  + {best_name:<26} -> macro {best_value:.4f}")

    baseline_name = rows[0]["run"]
    baseline = scores[baseline_name]
    print(f"\n== ensemble of {len(chosen)} vs best single ({baseline_name}) ==")
    print(f"  ensemble macro : {best_score:.4f}")
    print(f"  best single    : {rows[0]['macro']:.4f}")

    unique_patients = np.unique(patients)
    deltas = []
    for _ in range(args.reps):
        draw = RNG.choice(unique_patients, size=len(unique_patients), replace=True)
        index = np.concatenate([np.where(patients == p)[0] for p in draw])
        a = macro(targets[index], current[index])
        b = macro(targets[index], baseline[index])
        if a is not None and b is not None:
            deltas.append(a - b)
    deltas = np.array(deltas)
    print(
        f"  paired delta   : {deltas.mean():+.4f}  95% CI "
        f"[{np.percentile(deltas, 2.5):+.4f}, {np.percentile(deltas, 97.5):+.4f}]  "
        f"(n={len(deltas)})"
    )

    ensemble_metrics = summarize(targets, current, 0.0)
    mask10 = counts >= 10
    ensemble_10 = summarize(targets[mask10], current[mask10], 0.0)
    print("\n== per class, ensemble ==")
    print(f"{'cls':>4}{'pos':>6}{'AUC':>8}{'AUC(>=10)':>11}{'gap':>8}")
    profile = []
    for index, label in enumerate(CLASSES):
        auc = ensemble_metrics["per_class_auc"][index] or float("nan")
        auc10 = ensemble_10["per_class_auc"][index] or float("nan")
        print(f"{label:>4}{ensemble_metrics['positive_counts'][index]:>6}{auc:8.3f}{auc10:11.3f}{auc - auc10:8.3f}")
        profile.append({"class": label, "auc": auc, "auc_ge10": auc10})

    report = {
        "valid": str(args.valid),
        "studies": len(studies),
        "patients": len(unique_patients),
        "single_runs": rows,
        "ensemble": {
            "members": chosen,
            "macro": best_score,
            "macro_ge10": ensemble_10["macro_auc"],
            "per_class": profile,
            "curve": curve,
            "note": "greedy selection on official valid: optimistic, do not use as a generalisation estimate",
        },
        "bootstrap": {
            "baseline": baseline_name,
            "delta_mean": float(deltas.mean()),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "reps": int(len(deltas)),
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
