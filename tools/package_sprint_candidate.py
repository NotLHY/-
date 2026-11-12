"""Build an isolated B0 candidate with inference, model, training evidence and CSV.

This produces a draft archive. Official declaration workbook and T4 acceptance remain
explicit checks; no score or latency is inferred from generating a package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "release" / "p8_ens3_20261109"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or out.with_suffix(".zip").exists():
        raise FileExistsError(out)
    # Validate before creating output; the bundled predictor has exactly this architecture.
    for checkpoint in args.checkpoints:
        cfg = torch.load(checkpoint, map_location="cpu", weights_only=True)["config"]
        if (
            cfg["backbone"] != "efficientnet_b0"
            or cfg["pooling"] != "image_mean"
            or any(
                cfg.get(k)
                for k in ("view_conditioning", "orientation_conditioning", "tta", "clahe", "draft")
            )
            or cfg.get("max_eval_views") is not None
        ):
            raise ValueError(
                "Bundled predictor supports B0/image_mean/full decode/all views/no TTA only"
            )
        if not (checkpoint.parent / "code_snapshot" / "manifest.json").is_file():
            raise ValueError(f"Missing training code snapshot: {checkpoint}")
    (out / "model").mkdir(parents=True)
    (out / "code").mkdir()
    for filename in ("predict.py", "cost.py", "requirements.txt"):
        shutil.copy2(TEMPLATE / "code" / filename, out / "code" / filename)
    shutil.copy2(TEMPLATE / "cost.py", out / "cost.py")
    shutil.copytree(TEMPLATE / "licenses", out / "licenses")
    shutil.copytree(TEMPLATE / "pretrain_model", out / "pretrain_model")
    spec = json.loads((TEMPLATE / "model" / "deployment.json").read_text(encoding="utf-8"))
    spec["candidate_id"] = out.name
    spec.pop("created_at", None)
    spec["preprocessing"]["jpeg_draft"] = False
    spec["models"] = []
    count = len(args.checkpoints)
    spec["fusion"]["weights"] = [1.0 / count] * count
    declarations = []
    for index, checkpoint in enumerate(args.checkpoints):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = state["config"]
        target = out / "model" / f"member_{index}.pt"
        torch.save({"model": state["model"]}, target)
        spec["models"].append(
            {
                "model_id": f"member_{index}",
                "backbone": cfg["backbone"],
                "pooling": cfg["pooling"],
                "image_size": cfg["image_size"],
                "checkpoint": target.relative_to(out).as_posix(),
                "sha256": digest(target),
                "bytes": target.stat().st_size,
            }
        )
        evidence = out / "training" / str(index)
        evidence.mkdir(parents=True)
        shutil.copytree(checkpoint.parent / "code_snapshot", evidence / "code_snapshot")
        for filename in (
            "config.json",
            "environment.json",
            "split_manifest.json",
            "experiment_receipt.json",
            "best_metrics.json",
            "metrics.jsonl",
        ):
            source = checkpoint.parent / filename
            if source.is_file():
                shutil.copy2(source, evidence / filename)
        declarations.append(
            {
                "source_checkpoint_sha256": digest(checkpoint),
                "merged_official_valid": cfg.get("merge_valid", False),
                "validation_policy": cfg["validation"],
                "epoch": state["epoch"],
            }
        )
    if sum(p.stat().st_size for p in (out / "model").iterdir()) > 500_000_000:
        raise ValueError("Model files exceed 500 MB")
    (out / "model" / "deployment.json").write_text(
        json.dumps(spec, indent=2) + "\n", encoding="utf-8"
    )
    subprocess.run(
        [
            sys.executable,
            str(out / "code" / "predict.py"),
            "--data-root",
            str(args.data_root.resolve()),
            "--output",
            str(out / "submission.csv"),
            "--device",
            args.device,
        ],
        check=True,
    )
    (out / "release_status.json").write_text(
        json.dumps(
            {
                "status": "draft_pending_official_verification",
                "models": declarations,
                "submission_generated_by_bundled_predictor": True,
                "t4_verified": False,
                "official_declaration_workbook_completed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# Candidate draft\n\nEqual probability fusion; all ten classes exported.\n"
        "Training sources/configuration/environment are in training/.\n"
        "Before official delivery: complete the official declaration workbook; run\n"
        "`python cost.py submission.csv pictures_100 pictures_5 --data-root DATA`\n"
        "on a T4; keep its report and rebuild the archive after completing materials.\n"
        "This draft does not certify latency or a leaderboard score.\n",
        encoding="utf-8",
    )
    (out / "files.sha256").write_text(
        "".join(
            f"{digest(p)}  {p.relative_to(out).as_posix()}\n"
            for p in sorted(out.rglob("*"))
            if p.is_file()
        ),
        encoding="utf-8",
    )
    archive_path = out.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out).as_posix())
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{digest(archive_path)}  {archive_path.name}\n", encoding="utf-8"
    )
    print(archive_path)


if __name__ == "__main__":
    main()
