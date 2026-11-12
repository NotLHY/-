"""Focused tests for view metadata, test-view reweighting, and cache geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cxr.config import Config
from cxr.data import StudyDataset, _pad_resize, collate_studies, transform_image
from cxr.metrics import summarize
from cxr.records import ImageRecord, StudyRecord, study_view_group


def _image(path: Path, value: int = 80) -> ImageRecord:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (70, 90), value).save(path)
    return ImageRecord(path.stem, path, "", "")


def test_study_view_group_distinguishes_ap_and_lateral(tmp_path: Path) -> None:
    image = _image(tmp_path / "a.jpg")
    ap = ImageRecord(image.image_id, image.path, "AP", "")
    lateral = ImageRecord("b", tmp_path / "b.jpg", "LATERAL", "Recumbent")
    study = StudyRecord("s", "p", (ap, lateral), (1,))
    assert study_view_group(study) == "AP+lateral"
    only_ap = StudyRecord("s2", "p", (ap,), (1,))
    assert study_view_group(only_ap) == "AP only"


def test_reweighted_auc_follows_test_view_weights() -> None:
    targets = np.zeros((4, 10), dtype=np.float32)
    probabilities = np.zeros((4, 10), dtype=np.float32)
    for study in range(4):
        for label in range(10):
            positive = (study == 0 and label < 5) or (study == 1 and label >= 5)
            targets[study, label] = float(positive)
            base = 0.9 if positive else 0.1
            probabilities[study, label] = base if study < 2 else 1.0 - base
    metrics = summarize(
        targets,
        probabilities,
        0.0,
        view_groups=["A", "A", "B", "B"],
        view_weights={"A": 1.0, "B": 0.0},
    )
    assert metrics["macro_auc"] is not None
    assert metrics["macro_auc_reweighted"] == pytest.approx(1.0)
    assert metrics["view_group_auc"]["A"] == pytest.approx(1.0)


def test_cached_geometry_matches_direct_at_cache_size(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (70, 90), 123).save(source)
    with Image.open(source) as opened:
        cached = _pad_resize(opened.convert("L"), 64)
    cache_path = tmp_path / "source.png"
    cached.save(cache_path)
    direct = transform_image(str(source), 64, False, draft=False)
    cached_tensor = transform_image(str(source), 64, False, cache_path=cache_path, cache_size=64)
    assert np.allclose(direct.numpy(), cached_tensor.numpy(), atol=1e-6)


def test_collate_carries_orientation_and_mask(tmp_path: Path) -> None:
    image = _image(tmp_path / "s" / "a.jpg")
    image = ImageRecord(image.image_id, image.path, "AP", "Erect")
    study = StudyRecord("s", "p", (image,), (3,))
    dataset = StudyDataset([study], 64, training=False, cache_size=320, data_root=tmp_path)
    item = dataset[0]
    assert len(item) == 6
    batch = collate_studies([item])
    assert len(batch) == 7
    assert batch[1].sum().item() == 1
    assert batch[3][0, 0].item() == 1
    assert batch[4][0, 2].item() == 1.0


def test_config_defaults_preserve_old_behavior() -> None:
    config = Config(name="x", data_root=Path("."), output_dir=Path("."))
    assert config.view_conditioning is False
    assert config.orientation_conditioning is False
    assert config.warmup_epochs == 0
    assert config.grad_clip == 0.0
    assert config.max_eval_views is None
    assert config.reweight_views is False
    assert config.cache_size == 320
