"""Held-out longitudinal development set: patients excluded from every training run.

The official valid split is a random patient sample (3.8 checks per patient) while the
hidden test cohort is a follow-up cohort (10.4 checks per patient).  Judging models on
valid therefore overstates test performance by roughly five macro-AUC points, and its
frequent-visitor subset is too small (45 patients at >=5 checks, 20 at >=10) to resolve
a half-point difference.

A holdout manifest fixes that without touching test data: whole patients are carved out
of train and reserved for scoring only.  It is a JSON file with two lists:

``subjects``  every patient id to exclude from training
``studies``   every study id that belongs to those patients (the scoring scope)

Both lists are validated against the CSV at load time, so a stale manifest fails loudly
instead of silently training on held-out patients.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .records import StudyRecord


@dataclass(frozen=True)
class Holdout:
    path: Path
    subjects: frozenset[str]
    studies: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> Holdout:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: holdout manifest must be a JSON object")
        subjects = raw.get("subjects")
        studies = raw.get("studies")
        if not isinstance(subjects, list) or not subjects:
            raise ValueError(f"{path}: missing non-empty 'subjects' list")
        if not isinstance(studies, list) or not studies:
            raise ValueError(f"{path}: missing non-empty 'studies' list")
        if not all(isinstance(value, str) and value for value in subjects):
            raise ValueError(f"{path}: 'subjects' must contain non-empty strings")
        if not all(isinstance(value, str) and value for value in studies):
            raise ValueError(f"{path}: 'studies' must contain non-empty strings")
        if len(set(studies)) != len(studies):
            raise ValueError(f"{path}: duplicate study id in 'studies'")
        return cls(path.resolve(), frozenset(subjects), tuple(studies))

    def split_train(
        self, studies: list[StudyRecord]
    ) -> tuple[list[StudyRecord], list[StudyRecord]]:
        """Return ``(kept, dropped)``; dropped studies belong to held-out patients."""
        kept: list[StudyRecord] = []
        dropped: list[StudyRecord] = []
        for study in studies:
            (dropped if study.subject_id in self.subjects else kept).append(study)
        return kept, dropped

    def select(self, studies: list[StudyRecord]) -> list[StudyRecord]:
        """Return the held-out studies in manifest order, rejecting a partial match."""
        available = {study.study_id: study for study in studies}
        missing = [key for key in self.studies if key not in available]
        if missing:
            raise ValueError(
                f"Holdout manifest lists {len(missing)} studies absent from the split, "
                f"e.g. {missing[0]!r}"
            )
        return [available[key] for key in self.studies]

    def check_disjoint(self, other: list[StudyRecord], name: str) -> None:
        """Reject a manifest whose patients also appear in the evaluation split."""
        overlap = self.subjects & {study.subject_id for study in other}
        if overlap:
            raise ValueError(
                f"Holdout patients appear in {name}.csv ({len(overlap)} patients), "
                f"e.g. {sorted(overlap)[0]!r}"
            )
