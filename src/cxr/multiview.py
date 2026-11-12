"""Fixed study-level aggregation rules for multi-view studies (inference only).

No training, no test data. For a fixed checkpoint it scores every image once, then
compares three deterministic ways of turning per-image probabilities into one
probability per study: mean (current default), max, and logit-mean.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .artifacts import load_checkpoint, write_json
from .config import Config
from .data import transform_image
from .records import read_available
from .runtime import device_for

RULES = ("mean", "max", "logit_mean")
BATCH = 16


def _macro_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    values = [
        roc_auc_score(labels[:, j], scores[:, j])
        for j in range(10)
        if 0 < labels[:, j].sum() < len(labels)
    ]
    return float(np.mean(values)) if values else None


def sweep(config: Config, checkpoint: Path, split: str | None = None) -> dict[str, object]:
    device = device_for(config)
    data = read_available(config.data_root)
    name = split or "valid"
    studies = list(data[name].studies)
    model, _ = load_checkpoint(checkpoint, config, device)
    model.eval()

    flat = [(index, image) for index, study in enumerate(studies) for image in study.images]
    per_image = np.zeros((len(flat), 10), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(flat), BATCH):
            chunk = flat[start : start + BATCH]
            pixels = torch.stack(
                [
                    transform_image(
                        str(image.path),
                        config.image_size,
                        False,
                        draft=config.draft,
                        clahe=config.clahe,
                    )
                    for _, image in chunk
                ]
            ).unsqueeze(1)
            pixels = pixels.to(device)
            mask = torch.ones(pixels.shape[0], 1, dtype=torch.bool, device=device)
            codes = torch.zeros(pixels.shape[0], 1, dtype=torch.long, device=device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                _, scores = model(pixels, mask, codes, codes)
            per_image[start : start + len(chunk)] = scores.float().cpu().numpy()

    grouped: dict[int, list[int]] = defaultdict(list)
    for position, (index, _) in enumerate(flat):
        grouped[index].append(position)

    fused = {rule: np.zeros((len(studies), 10), dtype=np.float64) for rule in RULES}
    epsilon = 1e-6
    for index, positions in grouped.items():
        block = per_image[positions]
        fused["mean"][index] = block.mean(axis=0)
        fused["max"][index] = block.max(axis=0)
        clipped = np.clip(block, epsilon, 1 - epsilon)
        fused["logit_mean"][index] = 1 / (
            1 + np.exp(-np.log(clipped / (1 - clipped)).mean(axis=0))
        )

    labels = np.zeros((len(studies), 10), dtype=np.int8)
    for row, study in enumerate(studies):
        for label in study.labels or ():
            labels[row, label - 1] = 1
    subjects = np.array([study.subject_id for study in studies])
    counts = defaultdict(int)
    for subject in subjects:
        counts[subject] += 1
    n_studies = np.array([counts[s] for s in subjects])

    report: dict[str, object] = {
        "split": name,
        "checkpoint": str(checkpoint),
        "studies": len(studies),
        "images": len(flat),
        "single_image_studies": int((np.array([len(s.images) for s in studies]) == 1).sum()),
        "rules": {},
    }
    for rule in RULES:
        entry = {"all": _macro_auc(labels, fused[rule])}
        for threshold in (5, 8):
            keep = n_studies >= threshold
            if keep.sum() > 20:
                entry[f">={threshold}"] = _macro_auc(labels[keep], fused[rule][keep])
        report["rules"][rule] = entry  # type: ignore[index]

    write_json(config.output_dir / "multiview_sweep.json", report)
    return report
