"""Average several prediction files into one submission.

Equal-weight probability average over models. No weights are fitted, no test
labels are used, and the aggregation is deterministic, so it is the safest
ensemble form for the competition rules.

    python tools/merge_predictions.py \
        --inputs runs/p8/test_224.csv runs/p8/test_320.csv runs/p8/test_448.csv \
        --output submissions/p8_ens3.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load(path: Path) -> dict[tuple[str, int], float]:
    scores: dict[tuple[str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = {"Study_id", "Predict_class", "Probability"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            key = (row["Study_id"], int(row["Predict_class"]))
            if key in scores:
                raise ValueError(f"{path}: duplicate row {key}")
            value = float(row["Probability"])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{path}: probability outside [0,1] for {key}")
            scores[key] = value
    if not scores:
        raise ValueError(f"{path}: no rows")
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tables = [load(path) for path in args.inputs]
    keys = set(tables[0])
    for path, table in zip(args.inputs[1:], tables[1:]):
        if set(table) != keys:
            raise ValueError(f"{path}: study/class set differs from {args.inputs[0]}")
    expected = 10
    studies = {study for study, _ in keys}
    if any(sum(1 for s, _ in keys if s == study) != expected for study in studies):
        raise ValueError("every study must have exactly 10 classes")

    merged = {
        key: sum(table[key] for table in tables) / len(tables)
        for key in keys
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Study_id", "Predict_class", "Probability"])
        for study, label in sorted(keys):
            writer.writerow([study, label, f"{merged[(study, label)]:.10f}"])

    print(f"members : {len(tables)}")
    print(f"studies : {len(studies)}")
    print(f"rows    : {len(keys)}")
    print(f"output  : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
