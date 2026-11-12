"""Validated CSV boundary and immutable study records."""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, StrictStr

BASE_COLUMNS = {
    "Dicom_id",
    "Subject_id",
    "Study_id",
    "ViewPosition",
    "PatientOrientation",
    "Rows",
    "Columns",
}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class RawImage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    Dicom_id: StrictStr
    Subject_id: StrictStr
    Study_id: StrictStr
    ViewPosition: StrictStr
    PatientOrientation: StrictStr
    Rows: StrictStr
    Columns: StrictStr


class RawLabeledImage(RawImage):
    Predict_class: StrictStr


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: Path
    view: str
    orientation: str = ""


@dataclass(frozen=True)
class StudyRecord:
    study_id: str
    subject_id: str
    images: tuple[ImageRecord, ...]
    labels: tuple[int, ...] | None


def study_view_group(study: StudyRecord) -> str:
    """Collapse a study's image views into the groups used for test reweighting."""
    views = {image.view.strip().upper() for image in study.images}
    has_ap = any(view.startswith("AP") for view in views)
    has_pa = any(view.startswith("PA") for view in views)
    has_lateral = bool(views & {"LATERAL", "LL"})
    if has_ap and not has_pa:
        return "AP+lateral" if has_lateral else "AP only"
    if has_pa and not has_ap:
        return "PA+lateral" if has_lateral else "PA only"
    if has_ap and has_pa:
        return "AP+PA"
    return "lateral only" if has_lateral else "unknown"


@dataclass(frozen=True)
class SplitRecords:
    name: str
    studies: tuple[StudyRecord, ...]
    sha256: str


def _identifier(value: str, field: str, line: int) -> str:
    if not value or not SAFE_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Invalid {field} at CSV row {line}: {value!r}")
    return value


def _labels(value: str, line: int) -> tuple[int, ...]:
    if not value.strip():
        raise ValueError(f"Empty supervised labels at CSV row {line}")
    try:
        tokens = [token.strip() for token in value.split(";")]
        if not all(token.isdecimal() for token in tokens):
            raise ValueError("invalid label token")
        labels = tuple(sorted({int(token) for token in tokens}))
    except ValueError as error:
        raise ValueError(f"Invalid class at CSV row {line}: {value!r}") from error
    if not labels or any(label < 1 or label > 10 for label in labels):
        raise ValueError(f"Class outside 1..10 at CSV row {line}: {value!r}")
    if 10 in labels and len(labels) > 1:
        raise ValueError(f"No Finding overlaps abnormal class at CSV row {line}")
    return labels


def read_split(root: Path, split: str) -> SplitRecords:
    """Read one split.

    ``CXR_ALLOW_MISSING_IMAGES=1`` relaxes the per-image existence check for
    cache-only server runs: records still carry their original relative paths
    (used to locate cache files), but raw JPEGs need not be present.
    """
    csv_path = root / f"{split}.csv"
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    allow_missing = os.environ.get("CXR_ALLOW_MISSING_IMAGES", "").strip() == "1"
    grouped: dict[str, list[ImageRecord]] = {}
    subjects: dict[str, str] = {}
    labels: dict[str, tuple[int, ...] | None] = {}
    seen_images: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        required = BASE_COLUMNS | ({"Predict_class"} if split != "test" else set())
        if fields != required or len(reader.fieldnames or ()) != len(required):
            raise ValueError(f"{csv_path}: CSV columns must be exactly {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            if split == "test":
                RawImage.model_validate(row)
            else:
                RawLabeledImage.model_validate(row)
            image_id = _identifier(row["Dicom_id"], "Dicom_id", line)
            subject_id = _identifier(row["Subject_id"], "Subject_id", line)
            study_id = _identifier(row["Study_id"], "Study_id", line)
            if image_id in seen_images:
                raise ValueError(f"Duplicate Dicom_id {image_id} in {csv_path}")
            seen_images.add(image_id)
            try:
                if int(row["Rows"]) <= 0 or int(row["Columns"]) <= 0:
                    raise ValueError("nonpositive image dimensions")
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid dimensions at CSV row {line}") from error
            item_labels = None if split == "test" else _labels(row["Predict_class"], line)
            if study_id in subjects and (
                subjects[study_id] != subject_id or labels[study_id] != item_labels
            ):
                raise ValueError(f"Inconsistent subject or labels within study {study_id}")
            subjects[study_id], labels[study_id] = subject_id, item_labels
            image_path = root / split / subject_id / study_id / f"{image_id}.jpg"
            if not image_path.is_file() and not allow_missing:
                raise ValueError(f"Missing image: {image_path}")
            grouped.setdefault(study_id, []).append(
                ImageRecord(
                    image_id,
                    image_path,
                    row["ViewPosition"] or "",
                    row["PatientOrientation"] or "",
                )
            )
    if not grouped:
        raise ValueError(f"No studies in {csv_path}")
    studies = tuple(
        StudyRecord(key, subjects[key], tuple(images), labels[key])
        for key, images in grouped.items()
    )
    return SplitRecords(split, studies, digest)


def _auto_allow_missing() -> bool:
    """True when CXR_ALLOW_MISSING_IMAGES=1 or CXR_CACHE_ROOT is set.

    A pre-built cache (manifest.json present) makes the on-disk JPEG check
    unnecessary — the dataset pipeline reads PNGs from the cache directly.
    """
    import os

    if os.environ.get("CXR_ALLOW_MISSING_IMAGES", "").strip() in ("1", "true", "yes"):
        return True
    if os.environ.get("CXR_CACHE_ROOT", "").strip():
        return True
    return False


def read_available(
    root: Path,
    *,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    allow_missing_images: bool = False,
) -> dict[str, SplitRecords]:
    """Validate only the requested data scope; supervised callers exclude test.

    ``allow_missing_images`` skips the on-disk JPEG presence check and is only
    safe when every requested image is reachable via a pre-built cache. The flag
    is also auto-enabled when ``CXR_CACHE_ROOT`` is set in the environment.
    """
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"train", "valid", "test"}:
        raise ValueError("splits must contain distinct names from train, valid, test")
    allow_missing_images = allow_missing_images or _auto_allow_missing()
    found = {
        name: read_split(root, name, allow_missing=allow_missing_images)
        for name in splits
        if (root / f"{name}.csv").is_file()
    }
    image_ids: set[str] = set()
    for split in found.values():
        for study in split.studies:
            for image in study.images:
                if image.image_id in image_ids:
                    raise ValueError(f"Duplicate Dicom_id across splits: {image.image_id}")
                image_ids.add(image.image_id)
    for left_name, left in found.items():
        for right_name, right in found.items():
            if left_name >= right_name:
                continue
            patients = {study.subject_id for study in left.studies} & {
                study.subject_id for study in right.studies
            }
            studies = {study.study_id for study in left.studies} & {
                study.study_id for study in right.studies
            }
            if patients or studies:
                raise ValueError(
                    f"Overlap between {left_name} and {right_name}: "
                    f"patients={sorted(patients)[:5]}, studies={sorted(studies)[:5]}"
                )
    return found
