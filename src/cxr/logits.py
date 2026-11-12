"""Per-image logits: the raw material for post-hoc TTA, view weighting and ensembles.

The submission path collapses everything into one probability per study, so no
downstream decision (which flip to average, how to weight views, which members to
blend) can be re-taken without another forward pass.  A dump keeps the layer just
below that collapse: one row per (image, flip) carrying the per-class logits and the
pooling weight the model applied.

Reconstruction is exact for the linear-pooling families (``image_mean`` and the
plain/attention mean) because the head is linear::

    pooled_logits = sum_i w_i * view_logits_i,   sum_i w_i = 1

``image_mean`` averages *sigmoids* per view rather than the logits, and the dump
records enough to reproduce both conventions.  For ``study_attention`` the weights
themselves depend on the flip, so a flipped dump still re-pools with the weights of
the unflipped pass; treat attention dumps as approximate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np

from .data import VIEW_CODE, _eval_views
from .records import StudyRecord

CLASS_COUNT = 10

# pixels are (batch, views, channels, height, width)
FLIP_DIMS: dict[str, tuple[int, ...]] = {
    "none": (),
    "h": (-1,),
    "v": (-2,),
    "hv": (-2, -1),
}


def parse_flips(value: str) -> tuple[str, ...]:
    """Parse ``--dump-flips none,h`` into a validated, ordered, duplicate-free tuple."""
    names = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not names:
        raise ValueError("flip list is empty")
    unknown = [name for name in names if name not in FLIP_DIMS]
    if unknown:
        raise ValueError(f"unknown flips {unknown}; expected {sorted(FLIP_DIMS)}")
    if "none" not in names:
        raise ValueError("flip list must contain 'none'")
    if len(set(names)) != len(names):
        raise ValueError("duplicate flip in list")
    return names


def extra_flips(flips: Iterable[str]) -> tuple[str, ...]:
    return tuple(name for name in flips if name != "none")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values.astype(np.float64)))


@dataclass(frozen=True)
class Frames:
    """Flat image rows, before dicom ids are resolved against the split records."""

    studies: tuple[str, ...]
    image_study: np.ndarray
    view_codes: np.ndarray
    orientation_codes: np.ndarray
    weights: np.ndarray
    logits: dict[str, np.ndarray]
    dicom_ids: tuple[str, ...] = ()

    @property
    def image_count(self) -> int:
        return int(self.image_study.shape[0])


def assemble(
    batches: Iterable[
        tuple[
            list[str],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            dict[str, np.ndarray],
        ]
    ],
) -> Frames:
    """Merge per-batch tensors into flat image rows.

    Each batch is ``(study_ids, mask, view_codes, orientation_codes, weights,
    view_logits_by_flip)`` with ``mask``/``view_codes`` shaped ``(batch, views)`` and
    each ``view_logits`` entry shaped ``(batch, views, 10)``.
    """
    studies: list[str] = []
    image_study: list[np.ndarray] = []
    views: list[np.ndarray] = []
    orientations: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    logits: dict[str, list[np.ndarray]] = {}
    for study_ids, mask, view_codes, orientation_codes, batch_weights, per_flip in batches:
        mask = np.asarray(mask, dtype=bool)
        counts = mask.sum(axis=1)
        flat_mask = mask.reshape(-1)
        offset = len(studies)
        studies.extend(study_ids)
        image_study.append(np.repeat(np.arange(len(study_ids)) + offset, counts))
        views.append(np.asarray(view_codes).reshape(-1)[flat_mask].astype(np.int8))
        orientations.append(np.asarray(orientation_codes).reshape(-1)[flat_mask].astype(np.int8))
        weights.append(np.asarray(batch_weights, dtype=np.float64).reshape(-1)[flat_mask])
        for name, values in per_flip.items():
            stacked = np.asarray(values, dtype=np.float64).reshape(-1, CLASS_COUNT)
            logits.setdefault(name, []).append(stacked[flat_mask])
    if not studies:
        raise ValueError("no batches to assemble")
    merged = Frames(
        studies=tuple(studies),
        image_study=np.concatenate(image_study).astype(np.int32),
        view_codes=np.concatenate(views),
        orientation_codes=np.concatenate(orientations),
        weights=np.concatenate(weights).astype(np.float32),
        logits={name: np.concatenate(chunks) for name, chunks in logits.items()},
    )
    sizes = {name: values.shape[0] for name, values in merged.logits.items()}
    if len(set(sizes.values())) != 1 or next(iter(sizes.values())) != merged.image_count:
        raise ValueError(f"flip shapes disagree with image rows: {sizes}")
    return merged


def attach_records(
    frames: Frames, studies: list[StudyRecord], max_eval_views: int | None
) -> Frames:
    """Resolve dicom ids and prove the rows are the images the model actually saw.

    Cross-checks the recorded view codes against the split CSV, so a row/ordering
    mismatch fails loudly instead of silently poisoning every downstream experiment.
    """
    expected_study_ids = [study.study_id for study in studies]
    if list(frames.studies) != expected_study_ids:
        raise ValueError(
            f"dump studied {len(frames.studies)} studies in a different order than the split"
        )
    dicom_ids: list[str] = []
    expected_views: list[int] = []
    counts = np.bincount(frames.image_study, minlength=len(studies))
    for index, study in enumerate(studies):
        chosen = _eval_views(study.images, max_eval_views)
        if len(chosen) != int(counts[index]):
            raise ValueError(
                f"{study.study_id}: dumped {int(counts[index])} images, "
                f"but the loader selected {len(chosen)}"
            )
        dicom_ids.extend(image.image_id for image in chosen)
        expected_views.extend(VIEW_CODE.get(image.view.strip().upper(), 0) for image in chosen)
    recorded = frames.view_codes.astype(np.int64)
    if not np.array_equal(recorded, np.asarray(expected_views, dtype=np.int64)):
        raise ValueError("dumped view codes do not match the images the split selects")
    return Frames(
        studies=frames.studies,
        image_study=frames.image_study,
        view_codes=frames.view_codes,
        orientation_codes=frames.orientation_codes,
        weights=frames.weights,
        logits=frames.logits,
        dicom_ids=tuple(dicom_ids),
    )


def pooled_probabilities(frames: Frames, flip: str, pooling: str) -> np.ndarray:
    """Rebuild study probabilities from a dump, matching the model's own convention."""
    if flip not in frames.logits:
        raise ValueError(f"dump has no '{flip}' flip; available {sorted(frames.logits)}")
    per_image = frames.logits[flip]
    weights = frames.weights.astype(np.float64)
    study_count = len(frames.studies)
    weighted_logits = np.zeros((study_count, CLASS_COUNT), dtype=np.float64)
    weighted_probability = np.zeros((study_count, CLASS_COUNT), dtype=np.float64)
    np.add.at(weighted_logits, frames.image_study, per_image * weights[:, None])
    np.add.at(weighted_probability, frames.image_study, _sigmoid(per_image) * weights[:, None])
    if pooling == "image_mean":
        return weighted_probability
    return _sigmoid(weighted_logits)


def check_dump_against_scores(
    frames: Frames,
    probabilities: np.ndarray,
    pooling: str,
    flips: tuple[str, ...] = ("none",),
    tolerance: float = 2e-3,
) -> float:
    """Prove the dump rebuilds the probabilities the run just reported.

    The scored array is produced under autocast in the real run, so the comparison is
    bounded by fp16 precision rather than by exact float64 equality.  ``flips`` lists
    the variants the run itself averaged (config-level TTA), in the same order.
    """
    rebuilt = np.mean([pooled_probabilities(frames, flip, pooling) for flip in flips], axis=0)
    if rebuilt.shape != probabilities.shape:
        raise ValueError(
            f"dump rebuilds {rebuilt.shape} but the run scored {probabilities.shape}"
        )
    error = float(np.abs(rebuilt - probabilities).max())
    if not np.isfinite(error):
        raise ValueError("dump reconstruction produced nonfinite values")
    if error > tolerance:
        raise ValueError(
            f"dump cannot reproduce the scored probabilities: max |delta|={error:.3e} "
            f"> {tolerance:.1e}"
        )
    return error


def write_dump(path: Path | BinaryIO, frames: Frames, meta: dict[str, object]) -> Path | BinaryIO:
    """Write one self-describing npz: meta json plus the image rows.

    Accepts a path or an open binary stream, so the round trip is testable without disk.
    """
    if not frames.dicom_ids:
        raise ValueError("refusing to write a dump without dicom ids")
    if isinstance(path, Path):
        path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "meta": np.array(json.dumps(meta, sort_keys=True)),
        "studies": np.asarray(frames.studies),
        "dicom_ids": np.asarray(frames.dicom_ids),
        "image_study": frames.image_study.astype(np.int32),
        "view_codes": frames.view_codes.astype(np.int8),
        "orientation_codes": frames.orientation_codes.astype(np.int8),
        "weights": frames.weights.astype(np.float32),
    }
    for flip, values in frames.logits.items():
        arrays[f"logits__{flip}"] = values.astype(np.float32)
    np.savez_compressed(path, **arrays)
    return path


def read_dump(path: Path | BinaryIO) -> tuple[Frames, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        logits = {
            key[len("logits__") :]: data[key].astype(np.float64)
            for key in data.files
            if key.startswith("logits__")
        }
        frames = Frames(
            studies=tuple(str(value) for value in data["studies"]),
            image_study=data["image_study"].astype(np.int32),
            view_codes=data["view_codes"].astype(np.int8),
            orientation_codes=data["orientation_codes"].astype(np.int8),
            weights=data["weights"].astype(np.float32),
            logits=logits,
            dicom_ids=tuple(str(value) for value in data["dicom_ids"]),
        )
    if frames.image_count != len(frames.dicom_ids):
        raise ValueError(f"{path}: {frames.image_count} rows but {len(frames.dicom_ids)} dicom ids")
    return frames, meta


@dataclass
class LogitCollector:
    """Accumulates per-batch view logits while a split is scored."""

    flips: tuple[str, ...] = ("none",)
    _batches: list[
        tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]
    ] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.flips = parse_flips(",".join(self.flips))

    @property
    def extra(self) -> tuple[str, ...]:
        return extra_flips(self.flips)

    def add(
        self,
        study_ids: list[str],
        mask: np.ndarray,
        view_codes: np.ndarray,
        orientation_codes: np.ndarray,
        weights: np.ndarray,
        per_flip: dict[str, np.ndarray],
    ) -> None:
        missing = [name for name in self.flips if name not in per_flip]
        if missing:
            raise ValueError(f"collector missing flips {missing}")
        self._batches.append(
            (list(study_ids), mask, view_codes, orientation_codes, weights, dict(per_flip))
        )

    def frames(self) -> Frames:
        if not self._batches:
            raise ValueError("collector received no batches")
        return assemble(self._batches)
