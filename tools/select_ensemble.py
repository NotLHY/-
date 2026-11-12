"""Greedy rank-average ensemble selection with an honest holdout confirmation.

The point of this tool: model *selection* is where valid-split overfitting enters,
so selection and confirmation are kept on different splits.

    search   -> greedy Caruana forward selection on the dev split (e.g. the 2666-study
                internal train holdout shared by the seed-42 runs)
    confirm  -> score the chosen team on the official valid split and compare it to an
                incumbent team with a patient-cluster paired bootstrap
    in-sample-> the same greedy search run directly on the holdout, to measure how much
                of the apparent gain is selection bias

Only cached probability files are read; no model, no GPU.
"""

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


# ---------------------------------------------------------------- loading


def load_labels(csv_path: Path, keep_studies: set[str] | None = None) -> dict[str, set[int]]:
    """Study -> set of positive classes, from an image-level train/valid csv."""
    labels: dict[str, set[int]] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            study = row["Study_id"]
            if keep_studies is not None and study not in keep_studies:
                continue
            labels.setdefault(study, set()).update(
                int(token) for token in row["Predict_class"].split(";") if token
            )
    return labels


def load_patients(csv_path: Path, keep_studies: set[str] | None = None) -> dict[str, str]:
    subject: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            study = row["Study_id"]
            if keep_studies is not None and study not in keep_studies:
                continue
            subject[study] = row["Subject_id"]
    return subject


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    scores: dict[str, dict[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            study, label = row["Study_id"], int(row["Predict_class"])
            value = float(row["Probability"])
            if label not in CLASSES:
                raise ValueError(f"{path.name}: invalid class {label} for {study}")
            if not math.isfinite(value):
                raise ValueError(f"{path.name}: non-finite probability {study}/{label}")
            slot = scores.setdefault(study, {})
            if label in slot:
                raise ValueError(f"{path.name}: duplicate {study}/{label}")
            slot[label] = value
    matrix: dict[str, np.ndarray] = {}
    for study, slot in scores.items():
        if len(slot) != 10:
            raise ValueError(f"{path.name}: {study} has {len(slot)}/10 classes")
        matrix[study] = np.asarray([slot[c] for c in CLASSES], dtype=np.float64)
    return matrix


def stack(
    studies: list[str], labels: dict[str, set[int]], preds: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.zeros((len(studies), 10), dtype=np.int8)
    scores = np.zeros((len(studies), 10), dtype=np.float64)
    for i, study in enumerate(studies):
        for j, label in enumerate(CLASSES):
            truth[i, j] = int(label in labels[study])
            scores[i, j] = preds[study][j]
    return truth, scores


# ---------------------------------------------------------------- metrics


def macro_auc(truth: np.ndarray, scores: np.ndarray) -> tuple[float, list[float]]:
    metrics = summarize(truth, scores, 0.0)
    per_class = [v if v is not None else float("nan") for v in metrics["per_class_auc"]]
    defined = [v for v in per_class if not math.isnan(v)]
    value = float(np.mean(defined)) if len(defined) == 10 else float("nan")
    return value, per_class


def rank_transform(scores: np.ndarray) -> np.ndarray:
    """Per-class rank in (0,1). Monotone per class, so AUC is unchanged per member."""
    n = scores.shape[0]
    out = np.empty_like(scores)
    for j in range(scores.shape[1]):
        order = np.argsort(scores[:, j], kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        out[:, j] = ranks / max(n - 1, 1)
    return out


# ---------------------------------------------------------------- selection


def greedy(
    pool: dict[str, np.ndarray],
    truth: np.ndarray,
    max_members: int,
    eps: float,
    verbose: bool,
) -> tuple[list[str], list[float]]:
    """Caruana forward selection with replacement on a rank-average."""
    current: list[str] = []
    path: list[float] = []
    best = -math.inf
    for step in range(max_members):
        best_name, best_score = None, -math.inf
        for name, matrix in pool.items():
            team = current + [name]
            combined = np.mean([pool[m] for m in team], axis=0)
            value, _ = macro_auc(truth, combined)
            if value > best_score:
                best_name, best_score = name, value
        if best_name is None or best_score <= best + eps:
            break
        current.append(best_name)
        best = best_score
        path.append(best_score)
        if verbose:
            print(f"  step {step + 1}: +{best_name:<26s} dev macro={best_score:.4f}")
    return current, path


def team_scores(pool: dict[str, np.ndarray], team: list[str]) -> np.ndarray:
    return np.mean([pool[name] for name in team], axis=0)


# ---------------------------------------------------------------- paired CI


def paired_bootstrap(
    truth: np.ndarray,
    candidate: np.ndarray,
    incumbent: np.ndarray,
    patient_of: list[str],
    reps: int,
    seed: int,
) -> dict[str, object]:
    groups: dict[str, list[int]] = {}
    for index, patient in enumerate(patient_of):
        groups.setdefault(patient, []).append(index)
    patients = sorted(groups)
    rng = np.random.RandomState(seed)
    deltas: list[float] = []
    cand_only: list[float] = []
    inc_only: list[float] = []
    skipped = 0
    for _ in range(reps):
        picked = rng.choice(len(patients), size=len(patients), replace=True)
        idx = np.concatenate([groups[patients[i]] for i in picked])
        resample = truth[idx]
        positives = resample.sum(axis=0)
        if np.any(positives == 0) or np.any(positives == len(resample)):
            skipped += 1
            continue
        cand_macro, _ = macro_auc(resample, candidate[idx])
        inc_macro, _ = macro_auc(resample, incumbent[idx])
        if math.isnan(cand_macro) or math.isnan(inc_macro):
            skipped += 1
            continue
        deltas.append(cand_macro - inc_macro)
        cand_only.append(cand_macro)
        inc_only.append(inc_macro)
    array = np.asarray(deltas)
    ci = np.percentile(array, [2.5, 97.5]).tolist() if len(array) else [None, None]
    return {
        "reps_used": len(deltas),
        "reps_skipped": skipped,
        "delta_mean": float(array.mean()) if len(deltas) else None,
        "delta_ci95": ci,
        "p_delta_pos": float((array > 0).mean()) if len(deltas) else None,
        "candidate_mean": float(np.mean(cand_only)) if cand_only else None,
        "incumbent_mean": float(np.mean(inc_only)) if inc_only else None,
        "candidate_sd": float(np.std(cand_only)) if cand_only else None,
        "incumbent_sd": float(np.std(inc_only)) if inc_only else None,
        "delta_sd": float(array.std()) if len(deltas) else None,
    }


# ---------------------------------------------------------------- driver


def discover_runs(root: Path, pattern: str, studies: list[str]) -> dict[str, dict[str, np.ndarray]]:
    pool: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(root.glob(pattern)):
        name = path.parent.name
        try:
            preds = load_predictions(path)
        except ValueError as error:
            print(f"  skip {name}: {error}")
            continue
        if set(preds) != set(studies):
            print(f"  skip {name}: study set mismatch")
            continue
        pool[name] = preds
    return pool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--holdout-labels", type=Path, default=None, help="valid.csv")
    parser.add_argument(
        "--dev-manifest",
        type=Path,
        default=None,
        help="split_manifest.json whose valid_study_ids define the dev split",
    )
    parser.add_argument("--incumbent", default="ensemble_nocache")
    parser.add_argument("--max-members", type=int, default=8)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--output", type=Path, default=Path("_diag/ensemble_selection.json"))
    args = parser.parse_args()

    data_root = args.data_root or args.root.parent / "cxr_first" / "cxr_first"
    holdout_csv = args.holdout_labels or data_root / "valid.csv"
    train_csv = data_root / "train.csv"
    manifest = args.dev_manifest or args.root / "local_full_224" / "split_manifest.json"

    holdout_labels = load_labels(holdout_csv)
    holdout_patients = load_patients(holdout_csv)
    holdout_studies = sorted(holdout_labels)
    print(f"holdout: {len(holdout_studies)} studies, {len(set(holdout_patients.values()))} patients")

    holdout_pool = discover_runs(
        args.root, "*/evaluation*valid_predictions.csv", holdout_studies
    )
    print(f"holdout pool: {len(holdout_pool)} runs")

    dev_ids = set(json.loads(manifest.read_text(encoding="utf-8"))["valid_study_ids"])
    dev_labels = load_labels(train_csv, keep_studies=dev_ids)
    dev_patients = load_patients(train_csv, keep_studies=dev_ids)
    dev_studies = sorted(dev_labels)
    print(f"dev: {len(dev_studies)} studies, {len(set(dev_patients.values()))} patients")

    dev_pool = discover_runs(args.root, "*/validation_predictions.csv", dev_studies)
    print(f"dev pool: {len(dev_pool)} runs")
    shared = sorted(set(dev_pool) & set(holdout_pool))
    print(f"runs in both pools: {len(shared)}")

    dev_truth, _ = stack(dev_studies, dev_labels, dev_pool[shared[0]])
    dev_rank = {name: rank_transform(stack(dev_studies, dev_labels, dev_pool[name])[1]) for name in shared}

    hold_truth, _ = stack(holdout_studies, holdout_labels, holdout_pool[shared[0]])
    hold_rank = {
        name: rank_transform(stack(holdout_studies, holdout_labels, holdout_pool[name])[1])
        for name in holdout_pool
    }

    print("\nsingle-model dev / holdout macro-AUC")
    singles = []
    for name in shared:
        dev_value, _ = macro_auc(dev_truth, dev_rank[name])
        hold_value, _ = macro_auc(hold_truth, hold_rank[name])
        singles.append((name, dev_value, hold_value))
    for name, dev_value, hold_value in sorted(singles, key=lambda x: -x[2]):
        print(f"  {name:<28s} dev={dev_value:.4f}  holdout={hold_value:.4f}")

    print("\ngreedy selection on dev")
    team, path = greedy(dev_rank, dev_truth, args.max_members, 1e-5, verbose=True)
    print(f"  team: {team}")

    print("\nconfirmation on holdout")
    candidate = team_scores(hold_rank, team)
    incumbent = hold_rank[args.incumbent]
    cand_macro, cand_per = macro_auc(hold_truth, candidate)
    inc_macro, inc_per = macro_auc(hold_truth, incumbent)
    print(f"  incumbent {args.incumbent}: {inc_macro:.4f}")
    print(f"  candidate {len(team)}-member : {cand_macro:.4f}  delta={cand_macro - inc_macro:+.4f}")
    stats = paired_bootstrap(
        hold_truth, candidate, incumbent,
        [holdout_patients[s] for s in holdout_studies], args.reps, args.seed,
    )
    print(
        f"  paired bootstrap: delta_mean={stats['delta_mean']}, "
        f"CI95={stats['delta_ci95']}, P(delta>0)={stats['p_delta_pos']}, used={stats['reps_used']}"
    )
    if stats["delta_sd"]:
        print(
            f"  spread: candidate_sd={stats['candidate_sd']:.4f} "
            f"incumbent_sd={stats['incumbent_sd']:.4f} delta_sd={stats['delta_sd']:.4f} "
            f"-> pairing is {stats['candidate_sd'] / stats['delta_sd']:.2f}x narrower"
        )

    print("\nselection-bias probe: greedy run directly on the holdout")
    insample_team, insample_path = greedy(hold_rank, hold_truth, args.max_members, 1e-5, verbose=False)
    insample_macro, _ = macro_auc(hold_truth, team_scores(hold_rank, insample_team))
    print(f"  team={insample_team}")
    print(f"  in-sample holdout macro={insample_macro:.4f} (this number is optimistic)")

    result = {
        "holdout_studies": len(holdout_studies),
        "dev_studies": len(dev_studies),
        "singles": [
            {"run": n, "dev": d, "holdout": h} for n, d, h in sorted(singles, key=lambda x: -x[2])
        ],
        "team": team,
        "dev_path": path,
        "incumbent": args.incumbent,
        "incumbent_macro": inc_macro,
        "candidate_macro": cand_macro,
        "delta_macro": cand_macro - inc_macro,
        "paired": stats,
        "per_class_candidate": cand_per,
        "per_class_incumbent": inc_per,
        "per_class_delta": [c - i for c, i in zip(cand_per, inc_per)],
        "in_sample_team": insample_team,
        "in_sample_macro": insample_macro,
        "positive_counts": hold_truth.sum(axis=0).astype(int).tolist(),
        "patient_counts": Counter(holdout_patients.values()).most_common(3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
