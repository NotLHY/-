"""Run one isolated experiment and always assemble a small feedback archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr.codesnapshot import _collect, _git_info, _sha256  # noqa: E402
from cxr.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.validation not in {"holdout", "none"}:
        raise ValueError("Sprint runs require validation=holdout or none")
    if config.validation == "holdout" and args.holdout is None:
        raise ValueError("Development runs require --holdout")
    if config.validation == "none" and args.holdout is not None:
        raise ValueError("Frozen full-data training cannot take --holdout")
    git = _git_info(ROOT)
    if git.get("dirty") and not args.allow_dirty:
        raise ValueError("Commit code before experiments (or explicitly --allow-dirty)")
    run = config.output_dir
    if run.exists() and any(run.iterdir()) and not args.resume:
        raise FileExistsError(f"Experiment directory already exists: {run}")
    run.mkdir(parents=True, exist_ok=True)
    files, _ = _collect(ROOT)
    source_before = {key: _sha256(value) for key, value in files.items()}
    if args.resume:
        snapshot = json.loads((run / "code_snapshot" / "manifest.json").read_text(encoding="utf-8"))
        original = {item["path"]: item["sha256"] for item in snapshot["files"]}
        if original != source_before:
            raise ValueError(
                "Resume source differs from original snapshot; restore the run's source"
            )
    receipt = {
        "started_at": datetime.now(UTC).isoformat(),
        "git": git,
        "status": "running",
        "config_sha256": _sha256(args.config),
        "test_inference": False,
        "commands": [],
        "resume": args.resume,
    }
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))

    def execute(*command: str) -> None:
        receipt["commands"].append(list(command))
        with (run / "experiment.log").open("a", encoding="utf-8") as log:
            log.write(json.dumps(list(command)) + "\n")
            log.flush()
            process = subprocess.run(
                [sys.executable, "-m", "cxr", *command],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if process.returncode:
            raise RuntimeError(f"CLI failed ({process.returncode}); see {run / 'experiment.log'}")

    try:
        command = ["train", "--config", str(args.config.resolve())]
        if args.holdout is not None:
            command += ["--holdout-manifest", str(args.holdout.resolve())]
        if args.resume:
            command += ["--resume", str(run / "last.pt")]
        execute(*command)
        if config.validation == "holdout":
            checkpoint = run / "best.pt"
            common = [
                "evaluate",
                "--config",
                str(args.config.resolve()),
                "--checkpoint",
                str(checkpoint),
            ]
            execute(*common, "--split", "train", "--holdout-manifest", str(args.holdout.resolve()))
            if not config.merge_valid:
                execute(*common, "--split", "valid")
        else:
            checkpoint = run / "final.pt"
        receipt["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        }
        receipt["status"] = "completed"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        files, _ = _collect(ROOT)
        receipt["source_unchanged"] = source_before == {
            key: _sha256(value) for key, value in files.items()
        }
        if not receipt["source_unchanged"]:
            receipt["status"] = "source_changed"
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        (run / "experiment_receipt.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with zipfile.ZipFile(run / "feedback.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(run.iterdir()):
                if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".log", ".txt"}:
                    archive.write(path, path.name)
            snapshot = run / "code_snapshot"
            for path in sorted(snapshot.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(run).as_posix())
        digest = hashlib.sha256((run / "feedback.zip").read_bytes()).hexdigest()
        (run / "feedback.zip.sha256").write_text(f"{digest}  feedback.zip\n", encoding="utf-8")
        print(f"{receipt['status']}: {run / 'feedback.zip'}", flush=True)
    if receipt["status"] != "completed":
        raise RuntimeError("Source changed during experiment; review receipt before using results")


if __name__ == "__main__":
    main()
