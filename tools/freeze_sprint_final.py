"""Freeze a chosen dev run into an all-data, fixed-epoch final configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cxr.codesnapshot import _sha256  # noqa: E402
from cxr.config import Config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.selected_run.resolve()
    raw = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if raw["validation"] != "holdout" or raw.get("merge_valid"):
        raise ValueError("Select an independent dev run with official valid kept out of training")
    if (
        any(
            raw.get(k) is not None
            for k in (
                "init_from",
                "train_limit",
                "val_limit",
                "max_train_batches",
                "max_val_batches",
                "min_patient_studies",
            )
        )
        or not raw["pretrained"]
    ):
        raise ValueError("Finalization requires a complete from-pretrained development run")
    receipt = json.loads((run / "experiment_receipt.json").read_text(encoding="utf-8"))
    if receipt["status"] != "completed" or not receipt["source_unchanged"]:
        raise ValueError("Selected experiment must be completed with unchanged source")
    if receipt["checkpoint"]["sha256"] != _sha256(run / "best.pt"):
        raise ValueError("Selected checkpoint differs from completed experiment receipt")
    best = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
    if best["defined_auc_count"] != 10:
        raise ValueError("All ten dev AUCs must be defined")
    raw.update(
        name=raw["name"] + "_final",
        validation="none",
        merge_valid=True,
        schedule_epochs=raw.get("schedule_epochs") or raw["epochs"],
        epochs=best["epoch"],
        output_dir=str(ROOT / "runs" / "sprint_final" / (raw["name"] + "_final")),
        init_from=None,
    )
    Config.model_validate({**raw, **{k: Path(raw[k]) for k in ("data_root", "output_dir")}})
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    decision = {
        "selected_run": str(run),
        "selected_checkpoint_sha256": _sha256(run / "best.pt"),
        "fixed_epochs": best["epoch"],
        "schedule_epochs": raw["schedule_epochs"],
        "policy": "All choices frozen before train+valid; final.pt only; no validation selection",
    }
    args.output.with_suffix(".freeze.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
