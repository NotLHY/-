"""Latency and consistency check for the packaged CXR submission.

Official shape of the check: 100 pictures, 5 warm-up pictures, batch=1 serial, 3 rounds,
end-to-end including decoding, preprocessing, forward passes, fusion and writing the result.
Because one Study can hold several views, each picture is resolved to its Study and *all* of
that Study's views are read and timed; the report gives both per-picture and per-image time.

    python code/cost.py submission.csv pictures_100 pictures_5 --data-root DATA

Exit code 0 only when every round matches the submission (threshold, ranking, |dp| <= 1e-3).
On a non-T4 device the verdict is DIAGNOSTIC (exit 2), never an acceptance.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import (  # noqa: E402
    CLASS_IDS,
    StudyModel,
    image_path,
    load_deployment,
    load_rgb,
    read_studies,
    to_tensor,
)

ROOT = Path(__file__).resolve().parents[1]


def load_submission(path: Path) -> dict[tuple[str, int], float]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {
            (row["Study_id"], int(row["Predict_class"])): float(row["Probability"])
            for row in csv.DictReader(stream)
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("pictures_100", type=Path)
    parser.add_argument("pictures_5", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, default=ROOT / "model" / "deployment.json")
    parser.add_argument("--output", type=Path, default=ROOT / "cost_report.csv")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.rounds != 3:
        parser.error("正式核验要求连续 3 轮")
    spec = load_deployment(args.deployment)
    device = torch.device(args.device)
    studies, images = read_studies(args.data_root)
    reference = load_submission(args.submission)

    dicom_to_study = {d: s for s, ds in images.items() for d in ds}
    members = []
    for index, item in enumerate(spec["models"]):
        state = torch.load(ROOT / item["checkpoint"], map_location="cpu", weights_only=True)
        model = StudyModel()
        model.load_state_dict(state["model"])
        model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        members.append((model, int(item["image_size"]), float(spec["fusion"]["weights"][index])))

    def pictures(folder: Path, expected: int) -> list[Path]:
        items = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(items) != expected:
            raise SystemExit(f"{folder}: 需要恰好 {expected} 张，实际 {len(items)}")
        return items

    warmup = pictures(args.pictures_5, 5)
    timed = pictures(args.pictures_100, 100)

    def run_study(study: str) -> torch.Tensor:
        subject = studies[study]["subject"]
        decoded = [load_rgb(image_path(args.data_root, study, subject, d)) for d in images[study]]
        fused = torch.zeros(10, dtype=torch.float64)
        for model, size, weight in members:
            pixels = torch.stack([to_tensor(image, size) for image in decoded]).unsqueeze(0).to(device)
            mask = torch.ones(1, len(decoded), dtype=torch.bool, device=device)
            with torch.no_grad():
                fused += weight * model(pixels, mask)[0].double().cpu()
        return fused

    for path in warmup:
        run_study(dicom_to_study[path.stem])

    timings: dict[str, list[float]] = {p.stem: [] for p in timed}
    round_scores: list[dict[tuple[str, int], float]] = []
    views = 0
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        stream.write("round,Dicom_id,Study_id,ms\n")
        for round_id in range(args.rounds):
            scores: dict[tuple[str, int], float] = {}
            for path in timed:
                study = dicom_to_study[path.stem]
                start = time.perf_counter()
                probs = run_study(study)
                elapsed = (time.perf_counter() - start) * 1000.0
                timings[path.stem].append(elapsed)
                stream.write(f"{round_id},{path.stem},{study},{elapsed:.2f}\n")
                stream.flush()
                for class_id, score in zip(CLASS_IDS, probs.tolist(), strict=True):
                    scores[(study, class_id)] = score
            round_scores.append(scores)
            views = sum(len(images[dicom_to_study[p.stem]]) for p in timed)

    per_picture = {k: sum(v) / len(v) for k, v in timings.items()}
    mean_ms = sum(per_picture.values()) / len(per_picture)
    per_image_ms = mean_ms / (views / len(timed))

    worst = 0.0
    rank_bad = 0
    label_bad = 0
    for scores in round_scores:
        for key, value in scores.items():
            worst = max(worst, abs(value - reference[key]))
            if (value >= 0.5) != (reference[key] >= 0.5):
                label_bad += 1
        by_study: dict[str, list[int]] = {}
        for study, class_id in scores:
            by_study.setdefault(study, []).append(class_id)
        for study, class_ids in by_study.items():
            ordered = sorted(class_ids)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    a = (study, ordered[i])
                    b = (study, ordered[j])
                    if (reference[a] > reference[b]) != (scores[a] > scores[b]):
                        rank_bad += 1

    official = device.type == "cuda" and "T4" in torch.cuda.get_device_name().upper()
    verdict = "PASS" if (worst <= 1e-3 and label_bad == 0 and rank_bad == 0) else "FAIL"
    print(
        f"{'[CONCLUSION]' if official else '[DIAGNOSTIC]'} device={torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'} "
        f"pictures=100 views={views} rounds={args.rounds} "
        f"per-picture={mean_ms:.1f}ms per-image={per_image_ms:.1f}ms "
        f"max|dp|={worst:.2e} label_mismatch={label_bad} rank_mismatch={rank_bad} verdict={verdict}"
    )
    if verdict != "PASS":
        return 1
    if not official:
        print("[DIAGNOSTIC] 非 T4 设备，结果仅作诊断，不能作为验收依据")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
