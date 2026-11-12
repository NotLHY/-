"""Select and apply a multi-run ensemble using official-valid predictions.

Each candidate run must have ``evaluation_valid_predictions.csv`` (859 x 10) and,
unless ``--valid-only`` is used, a test ``submission.csv`` (1414 x 10). The script
exhaustively scores every non-empty subset and every enabled combination method on
the official valid split using raw macro-AUC; test metadata is never used for
selection. The winner is applied to test, and the final CSV gets Subject_id plus
an optional probability threshold for the official submission format.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cxr.metrics import summarize  # noqa: E402
from cxr.records import read_available, study_view_group  # noqa: E402


def _load(path: Path, ids: list[str]) -> np.ndarray:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            study = row["Study_id"]
            label = int(row["Predict_class"])
            if label < 1 or label > 10:
                raise ValueError(f"{path}: class outside 1..10")
            values.setdefault(study, [float("nan")] * 10)[label - 1] = float(row["Probability"])
    if set(values) != set(ids):
        raise ValueError(f"{path}: study ids differ from the reference split")
    matrix = np.asarray([values[study] for study in ids], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: missing class probabilities")
    return matrix


def _rank_average(matrices: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(matrices, axis=0)  # models x studies x classes
    count = stacked.shape[1]
    output = np.zeros((count, stacked.shape[2]), dtype=np.float64)
    for column in range(stacked.shape[2]):
        ranks = []
        for model in range(stacked.shape[0]):
            values = stacked[model, :, column]
            order = np.argsort(values, kind="stable")
            rank = np.empty(count, dtype=np.float64)
            rank[order] = np.arange(count, dtype=np.float64)
            sorted_values = values[order]
            start = 0
            while start < count:
                end = start + 1
                while end < count and sorted_values[end] == sorted_values[start]:
                    end += 1
                if end - start > 1:
                    rank[order[start:end]] = float(np.mean(np.arange(start, end)))
                start = end
            ranks.append(rank)
        output[:, column] = np.mean(ranks, axis=0) / max(1, count - 1)
    return output


def _combine(matrices: list[np.ndarray], method: str) -> np.ndarray:
    if method == "prob":
        return np.mean(np.stack(matrices, axis=0), axis=0)
    if method == "logit":
        probabilities = np.clip(np.stack(matrices, axis=0), 1e-6, 1 - 1e-6)
        logits = np.log(probabilities / (1 - probabilities))
        return 1.0 / (1.0 + np.exp(-np.mean(logits, axis=0)))
    if method == "rank":
        return _rank_average(matrices)
    raise ValueError(f"unknown method: {method}")


def _subset_score(
    method: str,
    indices: tuple[int, ...],
    valid_matrices: list[np.ndarray],
    truth: np.ndarray,
    groups: list[str],
) -> tuple[float, float, dict]:
    combined = _combine([valid_matrices[index] for index in indices], method)
    metrics = summarize(truth, combined, 0.0, groups, None)
    raw = metrics.get("macro_auc")
    primary = raw if raw is not None else -1.0
    return float(primary), float(raw if raw is not None else -1.0), metrics


def _write_predictions(
    path: Path,
    ids: list[str],
    matrix: np.ndarray,
    subjects: dict[str, str] | None = None,
    threshold: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = (["Subject_id"] if subjects is not None else []) + [
            "Study_id",
            "Predict_class",
            "Probability",
        ]
        writer.writerow(header)
        for study, scores in zip(ids, matrix, strict=True):
            for label, score in enumerate(scores, start=1):
                if threshold is not None and score < threshold:
                    continue
                row = [study, label, f"{float(score):.10f}"]
                if subjects is not None:
                    row.insert(0, subjects[study])
                writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--runs", required=True, help="comma-separated run directory names under runs/")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--methods", default="prob,logit")
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument(
        "--submission-threshold",
        type=float,
        default=0.5,
        help="Only write classes >= threshold to the final submission; 0 keeps all ten.",
    )
    args = parser.parse_args()
    if args.submission_threshold is not None and not 0 <= args.submission_threshold <= 1:
        parser.error("--submission-threshold must be in [0,1]")

    data = read_available(args.data_root.resolve())
    if "valid" not in data or "test" not in data:
        raise SystemExit("official valid and test splits are required")
    valid_studies = sorted(data["valid"].studies, key=lambda item: item.study_id)
    valid_ids = [study.study_id for study in valid_studies]
    test_studies = sorted(data["test"].studies, key=lambda item: item.study_id)
    test_ids = [study.study_id for study in test_studies]
    test_subjects = {study.study_id: study.subject_id for study in test_studies}
    truth = np.zeros((len(valid_ids), 10), dtype=np.float32)
    for row, study in enumerate(valid_studies):
        for label in study.labels or ():
            truth[row, label - 1] = 1.0
    groups = [study_view_group(study) for study in valid_studies]

    names = [name.strip() for name in args.runs.split(",") if name.strip()]
    valid_matrices: list[np.ndarray] = []
    test_matrices: list[np.ndarray] = []
    used: list[str] = []
    for name in names:
        run = ROOT / "runs" / name
        valid_path = run / "evaluation_valid_predictions.csv"
        test_path = run / "submission.csv"
        if not valid_path.is_file():
            print(f"skip {name}: missing {valid_path}", flush=True)
            continue
        try:
            valid_matrix = _load(valid_path, valid_ids)
        except ValueError as error:
            print(f"skip {name}: {error}", flush=True)
            continue
        test_matrix = None
        if not args.valid_only:
            if not test_path.is_file():
                print(f"skip {name}: missing {test_path}", flush=True)
                continue
            test_matrix = _load(test_path, test_ids)
        used.append(name)
        valid_matrices.append(valid_matrix)
        if test_matrix is not None:
            test_matrices.append(test_matrix)
    if not used:
        raise SystemExit("no usable runs")
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    print(f"runs={used} methods={methods}", flush=True)

    candidates: list[dict] = []
    best: dict | None = None
    for size in range(1, len(used) + 1):
        for indices in itertools.combinations(range(len(used)), size):
            for method in methods:
                primary, raw, metrics = _subset_score(
                    method, indices, valid_matrices, truth, groups
                )
                row = {
                    "runs": [used[index] for index in indices],
                    "method": method,
                    "macro_auc_reweighted": metrics.get("macro_auc_reweighted"),
                    "macro_auc": metrics.get("macro_auc"),
                    "primary": primary,
                    "raw": raw,
                    "per_class_auc_reweighted": metrics.get("per_class_auc_reweighted"),
                    "view_group_auc": metrics.get("view_group_auc"),
                }
                candidates.append(row)
                if best is None or (row["primary"], row["raw"]) > (best["primary"], best["raw"]):
                    best = row
    assert best is not None
    single_best = max(
        (row for row in candidates if len(row["runs"]) == 1),
        key=lambda row: (row["primary"], row["raw"]),
    )
    if len(best["runs"]) > 1 and best["primary"] - single_best["primary"] < args.min_gain:
        print(
            f"ensemble gain {best['primary'] - single_best['primary']:+.4f} "
            f"below --min-gain {args.min_gain:+.4f}; falling back to best single run",
            flush=True,
        )
        best = single_best
    print(
        f"best: runs={best['runs']} method={best['method']} "
        f"rw={best['macro_auc_reweighted']} raw={best['macro_auc']}",
        flush=True,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    best_indices = tuple(used.index(name) for name in best["runs"])
    output_payload = {
        "runs": used,
        "methods": methods,
        "best": best,
        "candidates": candidates,
        "valid_studies": len(valid_ids),
        "test_studies": len(test_ids),
        "view_weights": None,
    }
    (output / "ensemble_metrics.json").write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "ensemble_config.json").write_text(
        json.dumps(
            {
                "runs": best["runs"],
                "method": best["method"],
                "macro_auc_reweighted": best["macro_auc_reweighted"],
                "macro_auc": best["macro_auc"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "evaluation_ensemble_valid.json").write_text(
        json.dumps(
            {
                "macro_auc": best["macro_auc"],
                "macro_auc_reweighted": best["macro_auc_reweighted"],
                "runs": best["runs"],
                "method": best["method"],
                "per_class_auc_reweighted": best["per_class_auc_reweighted"],
                "view_group_auc": best["view_group_auc"],
                "study_count": len(valid_ids),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    combined_valid = _combine([valid_matrices[index] for index in best_indices], best["method"])
    _write_predictions(output / "evaluation_ensemble_valid_predictions.csv", valid_ids, combined_valid)
    if not args.valid_only:
        combined_test = _combine([test_matrices[index] for index in best_indices], best["method"])
        _write_predictions(
            output / "submission.csv",
            test_ids,
            combined_test,
            subjects=test_subjects,
            threshold=args.submission_threshold,
        )
        print(f"wrote {output / 'submission.csv'}", flush=True)
    print(f"wrote {output / 'ensemble_metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
