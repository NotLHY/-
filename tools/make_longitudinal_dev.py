"""Build a train-only longitudinal holdout manifest (the "mock test" ruler).

The hidden test cohort is a follow-up cohort: 136 patients, 10.4 checks each, 92% of
checks coming from patients with >=5 checks.  The official valid split is a random
patient sample with 3.8 checks per patient, so it overstates test performance and its
frequent-visitor subsets are too small to resolve a half-point difference.

This tool carves whole repeated-visit patients out of ``train.csv`` and writes a JSON
manifest that the trainer reads to exclude them and the evaluator reads to score them.
Nothing from test is used: no test labels, no test statistics, no test images.

Example::

    python tools/make_longitudinal_dev.py \
        --train-csv ../cxr_first/cxr_first/train.csv \
        --valid-csv ../cxr_first/cxr_first/valid.csv \
        --output ../configs/dev/longitudinal_200.json \
        --patients 200 --min-studies 5 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BUCKETS = ((5, 6, "5-6"), (7, 9, "7-9"), (10, 14, "10-14"), (15, 10**9, "15+"))


def read_studies(path: Path) -> tuple[dict[str, str], dict[str, set[int]]]:
    """Return ``(study -> subject, study -> labels)`` from a labeled CSV."""
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
    if not subject:
        raise ValueError(f"{path}: no studies")
    return subject, labels


def bucket_of(count: int) -> str:
    for low, high, name in BUCKETS:
        if low <= count <= high:
            return name
    return BUCKETS[-1][2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--valid-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patients", type=int, default=200)
    parser.add_argument("--min-studies", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subject, labels = read_studies(args.train_csv)
    valid_subject, _ = read_studies(args.valid_csv)

    studies_of: dict[str, list[str]] = defaultdict(list)
    for study, person in subject.items():
        studies_of[person].append(study)
    for person in studies_of:
        studies_of[person].sort()

    overlap = set(studies_of) & set(valid_subject.values())
    if overlap:
        raise ValueError(f"train and valid share {len(overlap)} patients")

    per_patient = {person: len(items) for person, items in studies_of.items()}
    pool = sorted(person for person, count in per_patient.items() if count >= args.min_studies)
    if len(pool) < args.patients:
        raise ValueError(f"donor pool has {len(pool)} patients, need {args.patients}")

    by_bucket: dict[str, list[str]] = defaultdict(list)
    for person in pool:
        by_bucket[bucket_of(per_patient[person])].append(person)

    rng = random.Random(args.seed)
    chosen: list[str] = []
    for name in sorted(by_bucket):
        members = sorted(by_bucket[name])
        share = len(members) / len(pool)
        take = max(1, round(share * args.patients))
        chosen.extend(rng.sample(members, min(take, len(members))))
    chosen = sorted(chosen[: args.patients])

    dev_studies = [study for person in chosen for study in studies_of[person]]
    dev_set = set(dev_studies)
    if len(dev_set) != len(dev_studies):
        raise ValueError("duplicate study in holdout")

    positive_studies = [
        sum(1 for study in dev_studies if label in labels[study]) for label in range(1, 11)
    ]
    positive_patients = [
        len(
            {
                subject[study]
                for study in dev_studies
                if label in labels[study]
            }
        )
        for label in range(1, 11)
    ]
    if min(positive_patients) < 20:
        raise ValueError(
            "holdout has a class with fewer than 20 positive patients: " f"{positive_patients}"
        )

    stats = {
        "train_studies": len(subject),
        "train_patients": len(per_patient),
        "donor_patients": len(pool),
        "donor_studies": sum(per_patient[person] for person in pool),
        "holdout_patients": len(chosen),
        "holdout_studies": len(dev_studies),
        "holdout_studies_per_patient": round(len(dev_studies) / len(chosen), 2),
        "remaining_train_studies": len(subject) - len(dev_studies),
        "holdout_positive_studies": positive_studies,
        "holdout_positive_patients": positive_patients,
        "bucket_patients": {name: sum(1 for p in chosen if bucket_of(per_patient[p]) == name)
                            for name in sorted(by_bucket)},
    }

    manifest = {
        "version": 1,
        "purpose": (
            "Repeated-visit patients carved out of train.csv to score models on a cohort "
            "shaped like the hidden test set. No test data, labels or statistics are used."
        ),
        "seed": args.seed,
        "min_studies": args.min_studies,
        "source": {
            "train_csv": str(args.train_csv.resolve()),
            "train_csv_sha256": hashlib.sha256(args.train_csv.read_bytes()).hexdigest(),
        },
        "subjects": chosen,
        "studies": dev_studies,
        "stats": stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.output}  subjects={len(chosen)} studies={len(dev_studies)}")


if __name__ == "__main__":
    main()
