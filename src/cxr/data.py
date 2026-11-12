"""Full-image transforms, optional 320px cache, and variable-view study batches."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from .records import ImageRecord, StudyRecord

VIEW_CODE = {"AP": 1, "PA": 2, "LATERAL": 3, "LL": 4, "PA LLD": 5}
ORIENT_CODE = {"": 0, "ERECT": 1, "RECUMBENT": 2, "SITTING": 3}
CACHE_SUFFIX = ".png"

StudyItem: TypeAlias = tuple[Tensor, Tensor, str, str, Tensor, Tensor]
StudyBatch: TypeAlias = tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, list[str], list[str]
]


def cache_root_from_env() -> Path | None:
    """Return the optional cache directory; the official server normally leaves it unset."""
    value = os.environ.get("CXR_CACHE_ROOT", "").strip()
    return Path(value).expanduser().resolve() if value else None


def cache_size_from_env(default: int = 320) -> int:
    value = os.environ.get("CXR_CACHE_SIZE", "").strip()
    return int(value) if value else default


def _pad_resize(image: Image.Image, size: int) -> Image.Image:
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


def _cache_path(cache_root: Path, data_root: Path, image: ImageRecord) -> Path:
    try:
        relative = image.path.relative_to(data_root)
    except ValueError as error:
        raise ValueError(f"Image is outside data_root: {image.path}") from error
    return cache_root / relative.with_suffix(CACHE_SUFFIX)


def _apply_clahe(image: Image.Image) -> Image.Image:
    """Local contrast equalisation of the luminance channel (portable-film compensation)."""
    import cv2
    import numpy as np

    grey = np.asarray(image.convert("L"), dtype=np.uint8)
    grid = max(1, min(8, grey.shape[0] // 32))
    equalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid, grid)).apply(grey)
    return Image.fromarray(equalised, mode="L").convert("RGB")


def transform_image(
    path: str,
    size: int,
    training: bool,
    cache_path: Path | None = None,
    cache_size: int = 320,
    draft: bool = True,
    clahe: bool = False,
    aug_level: str = "legacy",
) -> Tensor:
    """Decode one image and apply deterministic geometry; augmentation stays online.

    The cache stores a lossless 320px L-mode pad-resized image. The cache never contains
    augmentation, RGB expansion, or normalization, so those remain per-epoch / per-run choices.
    """
    if cache_path is not None:
        if cache_size < size:
            raise ValueError(f"Cache size {cache_size} is smaller than requested {size}")
        with Image.open(cache_path) as opened:
            image = ImageOps.exif_transpose(opened)
            image = image.convert("L")
            if cache_size != size:
                image = image.resize((size, size), Image.Resampling.BILINEAR)
            image = image.convert("RGB")
    else:
        with Image.open(path) as opened:
            is_jpeg = opened.format == "JPEG"
            if draft and is_jpeg:
                # libjpeg DCT scaling avoids decoding 3000px just to shrink to 224/320.
                opened.draft(opened.mode, (size, size))
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image = _pad_resize(image, size)
    if clahe:
        image = _apply_clahe(image)
    operations: list[object] = []
    if training:
        if aug_level == "strong":
            operations.extend(
                (
                    T.RandomAffine(
                        degrees=10,
                        translate=(0.08, 0.08),
                        scale=(0.9, 1.1),
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                    T.ColorJitter(brightness=0.15, contrast=0.15),
                )
            )
        else:
            operations.extend(
                (
                    T.RandomAffine(
                        degrees=5, translate=(0.03, 0.03), interpolation=InterpolationMode.BILINEAR
                    ),
                    T.ColorJitter(brightness=0.08, contrast=0.08),
                )
            )
    operations.extend((T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))))
    if training and aug_level == "strong":
        # After normalisation, so the erased patch is exactly zero in normalised space.
        operations.append(T.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0.0))
    return T.Compose(operations)(image)  # type: ignore[arg-type,return-value]


def _train_views(
    images: tuple[ImageRecord, ...], maximum: int, rng: random.Random
) -> list[ImageRecord]:
    if len(images) <= maximum:
        return list(images)
    frontal = [item for item in images if item.view.upper() in {"AP", "PA", "PA LLD"}]
    lateral = [item for item in images if item.view.upper() in {"LATERAL", "LL"}]
    chosen: list[ImageRecord] = []
    if frontal:
        chosen.append(rng.choice(frontal))
    if lateral and len(chosen) < maximum:
        chosen.append(rng.choice(lateral))
    remaining = [item for item in images if item not in chosen]
    rng.shuffle(remaining)
    return (chosen + remaining)[:maximum]


def _eval_views(images: tuple[ImageRecord, ...], maximum: int | None) -> list[ImageRecord]:
    if maximum is None or len(images) <= maximum:
        return list(images)
    frontal = [item for item in images if item.view.upper() in {"AP", "PA", "PA LLD"}]
    lateral = [item for item in images if item.view.upper() in {"LATERAL", "LL"}]
    chosen: list[ImageRecord] = []
    if frontal:
        chosen.append(frontal[0])
    if lateral and len(chosen) < maximum:
        chosen.append(lateral[0])
    for item in images:
        if len(chosen) >= maximum:
            break
        if item not in chosen:
            chosen.append(item)
    return chosen[:maximum]


class StudyDataset(Dataset[StudyItem]):
    def __init__(
        self,
        studies: Sequence[StudyRecord],
        size: int,
        training: bool = False,
        max_views: int = 2,
        seed: int = 42,
        cache_root: Path | None = None,
        cache_size: int = 320,
        max_eval_views: int | None = None,
        data_root: Path | None = None,
        draft: bool = True,
        clahe: bool = False,
        aug_level: str = "legacy",
    ) -> None:
        self.studies = list(studies)
        self.size = size
        self.training = training
        self.max_views = max_views
        self.max_eval_views = max_eval_views
        self.seed = seed
        self.cache_root = cache_root
        self.cache_size = cache_size
        self.data_root = data_root
        self.draft = draft
        self.clahe = clahe
        self.aug_level = aug_level
        self.epoch = 0
        if cache_root is not None and data_root is None:
            raise ValueError("cache_root requires data_root for relative cache paths")
        if cache_root is not None and cache_size < size:
            raise ValueError(f"Cache size {cache_size} is smaller than image_size {size}")
        if cache_root is not None:
            manifest_path = cache_root / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Cache manifest missing: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("partial"):
                raise ValueError(f"Refusing a partial cache: {manifest_path}")
            if int(manifest.get("size", -1)) != cache_size:
                raise ValueError(
                    f"CXR_CACHE_SIZE={cache_size} differs from cache size {manifest.get('size')}"
                )

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, index: int) -> StudyItem:
        study = self.studies[index]
        images = (
            _train_views(
                study.images,
                self.max_views,
                random.Random(self.seed + self.epoch * 1000003 + index),
            )
            if self.training
            else _eval_views(study.images, self.max_eval_views)
        )
        cached = [
            _cache_path(self.cache_root, self.data_root, image)
            if self.cache_root is not None and self.data_root is not None
            else None
            for image in images
        ]
        pixels = torch.stack(
            [
                transform_image(
                    str(image.path),
                    self.size,
                    self.training,
                    cache_path=cache,
                    cache_size=self.cache_size,
                    draft=self.draft,
                    clahe=self.clahe,
                    aug_level=self.aug_level,
                )
                for image, cache in zip(images, cached, strict=True)
            ]
        )
        targets = torch.zeros(10, dtype=torch.float32)
        for label in study.labels or ():
            targets[label - 1] = 1
        view_codes = torch.tensor(
            [VIEW_CODE.get(image.view.strip().upper(), 0) for image in images], dtype=torch.long
        )
        orientation_codes = torch.tensor(
            [
                ORIENT_CODE.get(image.orientation.strip().upper(), 0)
                for image in images
            ],
            dtype=torch.long,
        )
        return pixels, targets, study.study_id, study.subject_id, view_codes, orientation_codes


def collate_studies(items: list[StudyItem]) -> StudyBatch:
    count = len(items)
    max_views = max(item[0].shape[0] for item in items)
    _, channels, height, width = items[0][0].shape
    pixels = torch.zeros((count, max_views, channels, height, width), dtype=items[0][0].dtype)
    mask = torch.zeros((count, max_views), dtype=torch.bool)
    view_codes = torch.zeros((count, max_views), dtype=torch.long)
    orientation_codes = torch.zeros((count, max_views), dtype=torch.long)
    for index, item in enumerate(items):
        views = item[0].shape[0]
        pixels[index, :views] = item[0]
        mask[index, :views] = True
        view_codes[index, :views] = item[4]
        orientation_codes[index, :views] = item[5]
    return (
        pixels,
        mask,
        view_codes,
        orientation_codes,
        torch.stack([item[1] for item in items]),
        [item[2] for item in items],
        [item[3] for item in items],
    )
