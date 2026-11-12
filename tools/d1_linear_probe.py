"""D1 线性探针：冻结通用视觉表征（DINOv2-S / SigLIP）+ 逻辑回归线性头。

目标：判断自监督表征（D2 微调）值不值得投。若 valid 上 Macro-AUC >= 0.80，
才投 D2；否则放弃这条赌注，全力 B1（valid 并入终训）+ C2（弱类重加权）。

做法（标准线性探针）：
  1. 冻结 backbone，逐 study 提取 image_mean 特征；
  2. sklearn OneVsRestClassifier(LogisticRegression) 拟合 10 类；
  3. 在 valid 上算逐类 OVR ROC AUC + Macro-AUC。

用法（用本地 pytorch conda 环境，torch 2.10+cu128 / RTX 5060 Ti）：
  D:\\Program\\Anaconda\\envs\\pytorch\\python.exe tools/d1_linear_probe.py \
      --backbone vit_small_patch14_dinov2.lvd142m --size 224 --train-limit 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier

from cxr.data import _cache_path, _eval_views, cache_root_from_env, cache_size_from_env, transform_image
from cxr.records import StudyRecord, read_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D1 frozen-backbone linear probe")
    parser.add_argument("--root", default=None, help="data root; defaults to $CXR_DATA_ROOT or ../../cxr_first/cxr_first")
    parser.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--train-limit", type=int, default=None, help="subset studies for speed")
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--max-views", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def make_model(backbone: str, device: str) -> torch.nn.Module:
    model = timm.create_model(
        backbone, pretrained=True, num_classes=0, dynamic_img_size=True
    )
    model.eval().to(device)
    return model


def image_tensors(
    study: StudyRecord,
    size: int,
    max_views: int,
    device: str,
    data_root: Path | None = None,
    cache_root: Path | None = None,
    cache_size: int = 320,
) -> torch.Tensor:
    images = _eval_views(study.images, max_views) if max_views else list(study.images)
    kwargs: dict = {}
    if cache_root is not None and data_root is not None:
        cache_paths = [_cache_path(cache_root, data_root, img) for img in images]
    else:
        cache_paths = [None] * len(images)
    return torch.stack(
        [
            transform_image(
                str(image.path),
                size,
                training=False,
                cache_path=cp,
                cache_size=cache_size,
            )
            for image, cp in zip(images, cache_paths, strict=True)
        ]
    ).to(device)


def extract_features(
    model: torch.nn.Module,
    studies: list[StudyRecord],
    size: int,
    max_views: int,
    batch: int,
    device: str,
    data_root: Path | None = None,
    cache_root: Path | None = None,
    cache_size: int = 320,
) -> np.ndarray:
    """Study-level image_mean features: [len(studies), D]."""
    features = np.zeros((len(studies), 0))
    for start in range(0, len(studies), batch):
        chunk = studies[start : start + batch]
        # Collect all images in the chunk, tagged with their study index.
        buffers: list[torch.Tensor] = []
        counts: list[int] = []
        for study in chunk:
            tensors = image_tensors(
                study, size, max_views, device, data_root, cache_root, cache_size
            )
            buffers.append(tensors)
            counts.append(tensors.shape[0])
        block = torch.cat(buffers, dim=0)
        with torch.no_grad():
            out = model(block).cpu().numpy()
        if features.shape[1] == 0:
            features = np.zeros((len(studies), out.shape[1]), dtype=np.float32)
        # Average per study.
        cursor = 0
        for index, count in enumerate(counts):
            features[start + index] = out[cursor : cursor + count].mean(axis=0)
            cursor += count
        elapsed = time.time()
        print(f"  extracted {min(start + batch, len(studies))}/{len(studies)} studies", flush=True)
    return features


def labels_matrix(studies: list[StudyRecord]) -> np.ndarray:
    matrix = np.zeros((len(studies), 10), dtype=np.float32)
    for index, study in enumerate(studies):
        for label in study.labels or ():
            matrix[index, label - 1] = 1.0
    return matrix


def macro_auc(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, list[float | None]]:
    per_class: list[float | None] = []
    for index in range(10):
        column = targets[:, index]
        if np.unique(column).size < 2:
            per_class.append(None)
        else:
            per_class.append(float(roc_auc_score(column, probabilities[:, index])))
    defined = [value for value in per_class if value is not None]
    return float(np.mean(defined)), per_class


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[d1] backbone={args.backbone} size={args.size} device={device}", flush=True)

    if args.root:
        root = Path(args.root).expanduser().resolve()
    else:
        root = Path(os.environ.get("CXR_DATA_ROOT", "../../cxr_first/cxr_first")).expanduser().resolve()
    data = read_available(root, splits=("train", "valid"))
    train_studies = list(data["train"].studies)
    valid_studies = list(data["valid"].studies)
    if args.train_limit:
        train_studies = train_studies[: args.train_limit]
    if args.val_limit:
        valid_studies = valid_studies[: args.val_limit]
    print(
        f"[d1] train={len(train_studies)} studies, valid={len(valid_studies)} studies", flush=True
    )

    # Cache support (auto from env): CXR_CACHE_ROOT + CXR_CACHE_SIZE
    c_root = cache_root_from_env()
    c_size = cache_size_from_env(default=args.size)
    if c_root is not None:
        print(f"[d1] cache={c_root} size={c_size}px", flush=True)

    model = make_model(args.backbone, device)

    print("[d1] extracting train features ...", flush=True)
    started = time.time()
    x_train = extract_features(
        model, train_studies, args.size, args.max_views, args.batch, device,
        data_root=root, cache_root=c_root, cache_size=c_size,
    )
    y_train = labels_matrix(train_studies)
    print("[d1] extracting valid features ...", flush=True)
    x_valid = extract_features(
        model, valid_studies, args.size, args.max_views, args.batch, device,
        data_root=root, cache_root=c_root, cache_size=c_size,
    )
    y_valid = labels_matrix(valid_studies)
    print(f"[d1] feature dim={x_train.shape[1]} elapsed={time.time() - started:.0f}s", flush=True)

    print("[d1] fitting logistic head ...", flush=True)
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"), n_jobs=-1
    )
    clf.fit(x_train, y_train)
    raw = np.asarray(clf.predict_proba(x_valid))
    # 兼容两种返回：list of [N, 2]（转 [N, 10] 正类列）或已堆叠的 [N, 10]。
    proba = raw[:, :, 1].T if raw.ndim == 3 else raw

    macro, per_class = macro_auc(y_valid, proba)
    positive_counts = y_valid.sum(axis=0).astype(int).tolist()
    report = {
        "backbone": args.backbone,
        "size": args.size,
        "train_studies": len(train_studies),
        "valid_studies": len(valid_studies),
        "feature_dim": int(x_train.shape[1]),
        "macro_auc": macro,
        "per_class_auc": per_class,
        "positive_counts": positive_counts,
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[d1] saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
