"""Generate portable, fresh-run server recipes without changing local pilot configurations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cxr.config import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=("bce", "asl"), default="asl")
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-eval-views", type=int, default=2)
    parser.add_argument("--cache-size", type=int, default=320)
    for flag in (
        "view-conditioning",
        "orientation-conditioning",
        "channels-last",
        "cudnn-benchmark",
    ):
        parser.add_argument(
            f"--{flag}", action=argparse.BooleanOptionalAction, default=True
        )
    # Test-derived reweighting is forbidden for training/model selection; keep it opt-in only.
    parser.add_argument("--reweight-views", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    data = (args.data_root or Path(os.environ.get("CXR_DATA_ROOT", root / "cxr_first"))).resolve()
    destination = (args.destination or root / "configs" / "generated").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    common = {
        "loss": args.loss,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "num_workers": args.workers,
        "seed": args.seed,
        "train_limit": None,
        "val_limit": None,
        "validation": "internal",
        "cpu_threads": args.cpu_threads,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "view_conditioning": args.view_conditioning,
        "orientation_conditioning": args.orientation_conditioning,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip": args.grad_clip,
        "log_interval": args.log_interval,
        "max_eval_views": args.max_eval_views,
        "reweight_views": args.reweight_views,
        "channels_last": args.channels_last,
        "cudnn_benchmark": args.cudnn_benchmark,
        "cache_size": args.cache_size,
    }
    for pooling in ("image_mean", "study_mean"):
        name = f"server_{pooling}_{args.loss}_{args.image_size}_seed{args.seed}"
        output = root / "runs" / name
        config = Config(
            name=name,
            data_root=data,
            output_dir=output,
            pooling=pooling,
            **common,
        )
        raw = config.model_dump(mode="json")
        raw["data_root"] = os.path.relpath(data, destination)
        raw["output_dir"] = os.path.relpath(output, destination)
        path = destination / f"{name}.json"
        if path.exists():
            raise FileExistsError(f"Config exists: {path}; preserve it or choose a new seed/size")
        path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        print(path)
    smoke = Config(
        name="server_smoke",
        data_root=data,
        output_dir=root / "runs" / "server_smoke",
        image_size=args.image_size,
        batch_size=min(4, args.batch_size),
        epochs=1,
        num_workers=min(2, args.workers),
        train_limit=64,
        val_limit=32,
        max_train_batches=3,
        max_val_batches=2,
        seed=args.seed,
        cpu_threads=4,
        loss=args.loss,
    ).model_dump(mode="json")
    smoke["data_root"] = os.path.relpath(data, destination)
    smoke["output_dir"] = os.path.relpath(root / "runs" / "server_smoke", destination)
    smoke_path = destination / "server_smoke.json"
    if not smoke_path.exists():
        smoke_path.write_text(json.dumps(smoke, indent=2), encoding="utf-8")
        print(smoke_path)


if __name__ == "__main__":
    main()
