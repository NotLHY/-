"""Patient-disjoint validation and reproducible study selection."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .config import Config
from .holdout import Holdout
from .records import SplitRecords, StudyRecord, study_view_group


def test_view_weights(data: dict[str, SplitRecords]) -> dict[str, float] | None:
    """Target view-group distribution from test.csv; labels are never read."""
    if "test" not in data:
        return None
    counts = Counter(study_view_group(study) for study in data["test"].studies)
    total = sum(counts.values())
    if not total:
        return None
    return {group: count / total for group, count in sorted(counts.items())}


def _class_counts(studies: list[StudyRecord]) -> list[int]:
    return [
        sum(int(study.labels is not None and label in study.labels) for study in studies)
        for label in range(1, 11)
    ]


def _coverage_first(studies: list[StudyRecord], limit: int | None, seed: int) -> list[StudyRecord]:
    if limit is None or limit >= len(studies):
        return studies
    rng = random.Random(seed)
    candidates = studies.copy()
    rng.shuffle(candidates)
    chosen: list[StudyRecord] = []
    missing = set(range(1, 11))
    while candidates and missing and len(chosen) < limit:
        best = max(candidates, key=lambda study: len(set(study.labels or ()) & missing))
        if not (set(best.labels or ()) & missing):
            break
        chosen.append(best)
        candidates.remove(best)
        missing.difference_update(best.labels or ())
    chosen.extend(candidates[: limit - len(chosen)])
    return chosen


def choose_studies(
    config: Config, data: dict[str, SplitRecords], holdout: Holdout | None = None
) -> tuple[list[StudyRecord], list[StudyRecord], dict[str, object]]:
    all_train = list(data["train"].studies)
    if config.merge_valid:
        if "valid" not in data:
            raise ValueError("merge_valid requires valid.csv")
        all_train = all_train + list(data["valid"].studies)
    if config.validation == "official" and not config.merge_valid:
        if "valid" not in data:
            raise ValueError("Official validation requires valid.csv")
        train, valid = all_train, list(data["valid"].studies)
        source = "official"
    else:
        groups = [study.subject_id for study in all_train]
        if len(set(groups)) < 2:
            raise ValueError("Internal split needs at least two distinct patients")
        splitter = GroupShuffleSplit(
            n_splits=64, test_size=config.val_fraction, random_state=config.effective_split_seed
        )
        whole = np.asarray(_class_counts(all_train), dtype=float) / len(all_train)
        best: tuple[float, list[StudyRecord], list[StudyRecord]] | None = None
        for train_indices, val_indices in splitter.split(all_train, groups=groups):
            candidate_train = [all_train[int(i)] for i in train_indices]
            candidate_val = [all_train[int(i)] for i in val_indices]
            train_counts, val_counts = _class_counts(candidate_train), _class_counts(candidate_val)
            if any(x == 0 or x == len(candidate_train) for x in train_counts):
                continue
            if any(x == 0 or x == len(candidate_val) for x in val_counts):
                continue
            proportions = np.asarray(val_counts, dtype=float) / len(candidate_val)
            score = float(np.abs(whole - proportions).mean())
            if best is None or score < best[0]:
                best = (score, candidate_train, candidate_val)
        if best is None:
            raise ValueError(
                "Cannot create patient-disjoint partitions with both labels for every class"
            )
        _, train, valid = best
        source = "internal"
    holdout_info: dict[str, object] | None = None
    if holdout is not None:
        holdout.check_disjoint(valid, "the validation")
        train, dropped = holdout.split_train(train)
        if not train:
            raise ValueError("Holdout manifest removed every training study")
        holdout_info = {
            "manifest": str(holdout.path),
            "subjects": len(holdout.subjects),
            "excluded_studies": len(dropped),
            "scored_studies": len(holdout.studies),
        }
    specialist_info: dict[str, object] | None = None
    if config.min_patient_studies is not None:
        history = Counter(study.subject_id for study in all_train)
        eligible = {name for name, count in history.items() if count >= config.min_patient_studies}
        kept = [study for study in train if study.subject_id in eligible]
        if not kept:
            raise ValueError(
                f"No training patients have {config.min_patient_studies}+ studies after the holdout"
            )
        specialist_info = {
            "min_patient_studies": config.min_patient_studies,
            "eligible_patients": len(eligible),
            "kept_studies": len(kept),
            "dropped_studies": len(train) - len(kept),
        }
        train = kept
    full_valid_count = len(valid)
    train = _coverage_first(train, config.train_limit, config.seed)
    valid = _coverage_first(valid, config.val_limit, config.seed + 1)
    manifest: dict[str, object] = {
        "validation": source,
        "seed": config.seed,
        "split_seed": config.effective_split_seed,
        "csv_sha256": {name: part.sha256 for name, part in data.items()},
        "train_study_ids": [study.study_id for study in train],
        "train_subject_ids": sorted({study.subject_id for study in train}),
        "valid_study_ids": [study.study_id for study in valid],
        "valid_subject_ids": sorted({study.subject_id for study in valid}),
        "full_train_count": len(all_train),
        "full_valid_count": full_valid_count,
        "selected_class_counts": {"train": _class_counts(train), "valid": _class_counts(valid)},
        "test_view_weights": test_view_weights(data),
        "holdout": holdout_info,
        "specialist": specialist_info,
    }
    return train, valid, manifest


def manifest_studies(
    data: dict[str, SplitRecords], manifest: dict[str, object]
) -> list[StudyRecord]:
    source = "valid" if manifest["validation"] == "official" else "train"
    available = {study.study_id: study for study in data[source].studies}
    ids = manifest["valid_study_ids"]
    if not isinstance(ids, list):
        raise ValueError("Invalid split manifest")
    if set(ids) - available.keys():
        raise ValueError("Saved validation studies missing from current CSV")
    return [available[str(key)] for key in ids]
