"""Generate B1 configs (merge valid into train + internal validation) from the p8 recipe.

B1 = valid.csv 并入训练集终训。merge_valid 会把 train.csv + valid.csv 合并后做
内部 patient-disjoint 切分（val_fraction 0.05 早停），不再依赖官方 valid.csv。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE: dict[str, object] = {
    "data_root": "../../cxr_first/cxr_first",
    "backbone": "efficientnet_b0",
    "pretrained": True,
    "pooling": "image_mean",
    "loss": "asl",
    "epochs": 12,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "device": "cuda",
    "num_workers": 8,
    "max_train_views": 2,
    "train_limit": None,
    "val_limit": None,
    "val_fraction": 0.05,
    "validation": "internal",
    "merge_valid": True,
    "amp": True,
    "pin_memory": False,
    "freeze_backbone": False,
    "patience": 4,
    "cpu_threads": 4,
    "view_conditioning": False,
    "orientation_conditioning": False,
    "warmup_epochs": 1,
    "grad_clip": 1.0,
    "log_interval": 50,
    "channels_last": False,
    "cudnn_benchmark": True,
    "max_eval_views": None,
    "reweight_views": False,
    "patient_balanced": False,
    "tta": False,
    "draft": False,
    "aug_level": "legacy",
}

SIZES = [(224, 8, 320), (320, 8, 320), (448, 6, 448)]
SEEDS = [42, 43]


def main() -> None:
    out_dir = ROOT / "configs" / "p8b1"
    out_dir.mkdir(parents=True, exist_ok=True)
    for size, batch_size, cache_size in SIZES:
        for seed in SEEDS:
            config = dict(BASE)
            config.update(
                name=f"p8b1_b0_{size}_12ep_s{seed}",
                output_dir=f"../../runs/p8b1/p8b1_b0_{size}_12ep_s{seed}",
                image_size=size,
                batch_size=batch_size,
                cache_size=cache_size,
                seed=seed,
                split_seed=seed,
            )
            path = out_dir / f"p8b1_b0_{size}_12ep_s{seed}.json"
            path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
