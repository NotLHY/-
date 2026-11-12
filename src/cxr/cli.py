"""Public command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .holdout import Holdout
from .inference import audit, benchmark, evaluate, predict
from .multiview import sweep
from .runtime import train


def main() -> None:
    parser = argparse.ArgumentParser(prog="cxr")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "train", "evaluate", "predict", "benchmark", "multiview"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True, type=Path)
        if name in {"audit", "evaluate", "predict", "multiview"}:
            sub.add_argument(
                "--split", choices=("train", "valid", "test"), required=name == "predict"
            )
        if name in {"evaluate", "predict", "benchmark", "multiview"}:
            sub.add_argument("--checkpoint", required=True, type=Path)
        if name == "train":
            sub.add_argument("--resume", type=Path)
        if name in {"train", "evaluate"}:
            sub.add_argument(
                "--holdout-manifest",
                type=Path,
                help="JSON of held-out patients: excluded from train, scored by evaluate",
            )
        if name == "predict":
            sub.add_argument("--output", required=True, type=Path)
            sub.add_argument("--include-subject", action="store_true")
            sub.add_argument("--threshold", type=float)
        if name in {"evaluate", "predict"}:
            sub.add_argument(
                "--dump-logits",
                type=Path,
                help="write per-image logits + pooling weights so TTA and view "
                "weighting can be decided later without another forward pass",
            )
            sub.add_argument(
                "--dump-flips",
                default="none",
                help="comma separated flips to record, e.g. 'none,h'; 'none' is required",
            )
        if name == "benchmark":
            sub.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = getattr(args, "holdout_manifest", None)
    holdout = Holdout.load(manifest_path) if manifest_path is not None else None
    result: object
    if args.command == "audit":
        result = audit(config, args.split)
    elif args.command == "train":
        train(config, args.resume, holdout)
        return
    elif args.command == "evaluate":
        result = evaluate(
            config,
            args.checkpoint,
            args.split,
            dump_logits=args.dump_logits,
            dump_flips=args.dump_flips,
            holdout=holdout,
        )
    elif args.command == "predict":
        result = predict(
            config,
            args.checkpoint,
            args.split,
            args.output,
            args.include_subject,
            args.threshold,
            dump_logits=args.dump_logits,
            dump_flips=args.dump_flips,
        )
    elif args.command == "multiview":
        result = sweep(config, args.checkpoint, args.split)
    else:
        result = benchmark(config, args.checkpoint, args.repeats)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
