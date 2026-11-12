"""Read train/valid metadata only; produce reproducible study-level diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cxr.holdout import Holdout
from cxr.records import StudyRecord, read_available, study_view_group


def describe(studies: list[StudyRecord]) -> dict:
    history = Counter(s.subject_id for s in studies)
    return {
        "studies": len(studies),
        "patients": len(history),
        "images": sum(len(s.images) for s in studies),
        "max_studies_per_patient": max(history.values(), default=0),
        "view_counts": dict(sorted(Counter(len(s.images) for s in studies).items())),
        "view_groups": dict(sorted(Counter(study_view_group(s) for s in studies).items())),
        "classes": [
            {
                "id": c,
                "positive_studies": sum(c in (s.labels or ()) for s in studies),
                "positive_patients": len({s.subject_id for s in studies if c in (s.labels or ())}),
                "positive_single_view": sum(
                    c in (s.labels or ()) and len(s.images) == 1 for s in studies
                ),
                "largest_patient_positive_count": max(
                    Counter(s.subject_id for s in studies if c in (s.labels or ())).values(),
                    default=0,
                ),
            }
            for c in range(1, 11)
        ],
        "cooccurrence": [
            [
                sum(a in (s.labels or ()) and b in (s.labels or ()) for s in studies)
                for b in range(1, 11)
            ]
            for a in range(1, 11)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, default=Path("docs/splits/longitudinal_200.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = read_available(args.data_root, splits=("train", "valid"), allow_missing_images=True)
    holdout = Holdout.load(args.holdout)
    train = list(data["train"].studies)
    kept, _ = holdout.split_train(train)
    dev = holdout.select(train)
    report = {
        "scope": "train/valid CSV only; no test reads; no chronology inferred from opaque IDs",
        "csv_sha256": {name: part.sha256 for name, part in data.items()},
        "train": describe(train),
        "train_without_dev": describe(kept),
        "dev": describe(dev),
        "valid": describe(list(data["valid"].studies)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
