"""Deployment predictor for the CXR submission.

Reads deployment.json, then for every Study in <data-root>/test.csv:
  * loads every view at each member's image size (aspect-preserving pad-resize, bilinear),
  * runs each member (EfficientNet-B0 + linear head, image_mean pooling),
  * averages the members' study probabilities with fixed equal weights.

No test statistics, no fitting, no TTA: the same frozen weights always give the same CSV.
Only torch / torchvision / Pillow / numpy are required.

    python code/predict.py --data-root DATA --output submission.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms as T
from torchvision.models import efficientnet_b0

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASS_IDS = tuple(range(1, 11))
ROOT = Path(__file__).resolve().parents[1]


def load_deployment(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if tuple(spec["class_ids"]) != CLASS_IDS:
        raise ValueError("deployment.json: class_ids must be 1..10")
    weights = [float(w) for w in spec["fusion"]["weights"]]
    if len(weights) != len(spec["models"]) or abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError("deployment.json: fusion weights must match models and sum to 1")
    return spec


def pad_resize(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = min(size / width, size / height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR
    )
    left = (size - resized.width) // 2
    top = (size - resized.height) // 2
    return ImageOps.expand(
        resized,
        border=(left, top, size - resized.width - left, size - resized.height - top),
        fill=0,
    )


def load_rgb(path: Path) -> Image.Image:
    """Full-resolution decode (the trained configs use draft=false, so no DCT scaling)."""
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])(pad_resize(image, size))


class StudyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        network = efficientnet_b0(weights=None)
        self.encoder = nn.Sequential(network.features, network.avgpool, nn.Flatten())
        self.classifier = nn.Linear(1280, 10)

    def forward(self, pixels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, views, channels, height, width = pixels.shape
        flat = pixels.reshape(batch * views, channels, height, width)
        flat_mask = mask.reshape(-1)
        encoded = self.encoder(flat[flat_mask])
        features = encoded.new_zeros((batch * views, encoded.shape[-1]))
        features[flat_mask] = encoded
        features = features.reshape(batch, views, -1)
        logits = self.classifier(features)
        weights = mask.to(logits.dtype) / mask.sum(dim=1, keepdim=True)
        return (torch.sigmoid(logits) * weights.unsqueeze(-1)).sum(dim=1)


def read_studies(data_root: Path) -> tuple[OrderedDict[str, dict], dict[str, list[str]]]:
    studies: OrderedDict[str, dict] = OrderedDict()
    images: dict[str, list[str]] = {}
    with (data_root / "test.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            study = row["Study_id"]
            studies.setdefault(study, {"subject": row["Subject_id"]})
            images.setdefault(study, []).append(row["Dicom_id"])
    return studies, images


def image_path(data_root: Path, study: str, subject: str, dicom: str) -> Path:
    path = data_root / "test" / subject / study / f"{dicom}.jpg"
    if not path.is_file():
        raise FileNotFoundError(f"missing image: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", type=Path, default=ROOT / "model" / "deployment.json")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="smoke test: first N studies")
    args = parser.parse_args()

    spec = load_deployment(args.deployment)
    device = torch.device(args.device)
    studies, images = read_studies(args.data_root)
    names = list(studies)[: args.limit] if args.limit else list(studies)

    members = []
    for item in spec["models"]:
        path = ROOT / item["checkpoint"]
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = StudyModel()
        model.load_state_dict(state["model"])
        model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        members.append((model, int(item["image_size"]), float(spec["fusion"]["weights"][len(members)])))

    rows = []
    with torch.no_grad():
        for index, study in enumerate(names, start=1):
            subject = studies[study]["subject"]
            paths = [image_path(args.data_root, study, subject, d) for d in images[study]]
            decoded = [load_rgb(p) for p in paths]
            fused = torch.zeros(10, dtype=torch.float64)
            for model, size, weight in members:
                pixels = torch.stack([to_tensor(image, size) for image in decoded]).unsqueeze(0).to(device)
                mask = torch.ones(1, len(paths), dtype=torch.bool, device=device)
                fused += weight * model(pixels, mask)[0].double().cpu()
            for class_id, score in zip(CLASS_IDS, fused.tolist(), strict=True):
                rows.append((study, class_id, score))
            if index % 200 == 0:
                print(f"  {index}/{len(names)} studies", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["Study_id", "Predict_class", "Probability"])
        for study, class_id, score in rows:
            writer.writerow([study, class_id, f"{score:.10f}"])
    print(f"wrote {args.output} studies={len(names)} rows={len(rows)}")


if __name__ == "__main__":
    main()
