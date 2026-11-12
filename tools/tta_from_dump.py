"""Score flip / view recipes straight from a logit dump, with paired intervals.

This is the consumer the dump exists for: no model, no GPU, no new forward pass.
Every recipe here re-uses the *same* per-image logits, so the comparison is paired on
the same 859 studies and the interval reflects only evaluation noise.

    python tools/tta_from_dump.py --dump runs/_dumps/local_full_asl_valid.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from cxr.logits import pooled_probabilities, read_dump  # noqa: E402
from select_ensemble import (  # noqa: E402
    load_labels,
    load_patients,
    macro_auc,
    paired_bootstrap,
)


def rank_matrix(scores: np.ndarray) -> np.ndarray:
    """Per-class rank in (0,1): a monotone map, so per-class AUC is unchanged."""
    output = np.empty_like(scores)
    for column in range(scores.shape[1]):
        order = np.argsort(scores[:, column], kind="stable")
        ranks = np.empty(scores.shape[0])
        ranks[order] = np.arange(scores.shape[0])
        output[:, column] = ranks / max(scores.shape[0] - 1, 1)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--valid", type=Path, default=ROOT / "cxr_first" / "cxr_first" / "valid.csv")
    parser.add_argument("--reference", default="none")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    frames, meta = read_dump(args.dump)
    pooling = str(meta["pooling"])
    labels = load_labels(args.valid)
    patients = load_patients(args.valid)
    order = list(frames.studies)
    missing = [study for study in order if study not in labels]
    if missing:
        raise ValueError(f"{len(missing)} dumped studies are absent from {args.valid.name}")
    truth = np.array(
        [[int(number in labels[study]) for number in range(1, 11)] for study in order],
        dtype=np.int8,
    )
    patient_of = [patients[study] for study in order]

    print(f"dump: {args.dump.name} flip(s)={sorted(frames.logits)} pooling={pooling}")
    print(f"studies={len(order)} images={frames.image_count} patients={len(set(patient_of))}")
    print(f"reconstruction_error={meta.get('reconstruction_error')}\n")

    recipes: dict[str, np.ndarray] = {}
    for flip in frames.logits:
        recipes[flip] = pooled_probabilities(frames, flip, pooling)
    if len(frames.logits) > 1:
        recipes["mean(all flips)"] = np.mean(list(recipes.values()), axis=0)
        recipes["rank(all flips)"] = rank_matrix(np.mean(list(recipes.values()), axis=0))

    if args.reference not in recipes:
        raise ValueError(f"reference {args.reference!r} not in {sorted(recipes)}")

    reference_macro, _ = macro_auc(truth, recipes[args.reference])
    print(f"{'recipe':<20s} {'macro':>7s} {'delta':>8s} {'CI95':>20s} {'P(d>0)':>7s} verdict")
    print("-" * 78)
    rows = []
    for name, scores in recipes.items():
        value, _ = macro_auc(truth, scores)
        if name == args.reference:
            print(f"{name:<20s} {value:7.4f} {'—':>8s} {'—':>20s} {'—':>7s} reference")
            rows.append({"recipe": name, "macro": value, "delta": None})
            continue
        stats = paired_bootstrap(
            truth, scores, recipes[args.reference], patient_of, args.reps, args.seed
        )
        low, high = stats["delta_ci95"]
        verdict = "better" if low > 0 else "worse" if high < 0 else "undecided"
        print(
            f"{name:<20s} {value:7.4f} {value - reference_macro:+8.4f} "
            f"[{low:+.4f}, {high:+.4f}] {stats['p_delta_pos']:7.3f} {verdict}"
        )
        rows.append(
            {
                "recipe": name,
                "macro": value,
                "delta": value - reference_macro,
                "ci95": [low, high],
                "p_delta_pos": stats["p_delta_pos"],
                "verdict": verdict,
            }
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"dump": str(args.dump), "meta": meta, "reference": args.reference, "rows": rows},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
