"""The logit dump must rebuild exactly what the scoring path reported.

No GPU, no dataset: a tiny random-weight model plus fake records exercise the whole
chain (head -> collector -> frames -> npz -> pooled probabilities).  If this test ever
fails, every post-hoc TTA / view-weighting / ensemble decision made from a dump is
measuring something other than the model.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch

from cxr.config import Config
from cxr.logits import (
    LogitCollector,
    attach_records,
    parse_flips,
    pooled_probabilities,
    read_dump,
    write_dump,
)
from cxr.model import StudyModel
from cxr.records import ImageRecord, StudyRecord

PLACEHOLDER = Path("D:/nonexistent/logits")
POOLINGS = ("image_mean", "study_mean", "study_attention")
# A randomly initialised EfficientNet-B0 turns out to be flip-invariant on noise (its
# pooled features stop depending on the image at all), which would make the flip
# assertions vacuous. ResNet-18 keeps a real dependence on the pixels.
BACKBONE = "resnet18"


def make_config(**overrides: object) -> Config:
    raw: dict[str, object] = {
        "name": "logits_smoke",
        "data_root": PLACEHOLDER / "data",
        "output_dir": PLACEHOLDER / "out",
        "backbone": BACKBONE,
        "pretrained": False,
        "image_size": 64,
        "batch_size": 1,
        "epochs": 1,
        "device": "cpu",
        "num_workers": 0,
        "cache_size": 320,
    }
    raw.update(overrides)
    return Config.model_validate(raw)


def fake_studies(views: list[list[str]]) -> list[StudyRecord]:
    """StudyRecords with only the fields attach_records reads; no image files."""
    studies: list[StudyRecord] = []
    for study_index, view_list in enumerate(views):
        images = tuple(
            ImageRecord(
                image_id=f"pic_{study_index:03d}_{slot:02d}",
                path=PLACEHOLDER / f"{study_index}_{slot}.jpg",
                view=view,
            )
            for slot, view in enumerate(view_list)
        )
        studies.append(StudyRecord(f"study_{study_index:05d}", "subject_1", images, None))
    return studies


def batch_parts(
    model: StudyModel, studies: list[StudyRecord], flips: tuple[str, ...], seed: int = 0
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Run one batch and return the arrays a collector would receive."""
    generator = torch.Generator().manual_seed(seed)
    views = max(len(study.images) for study in studies)
    pixels = torch.randn(len(studies), views, 3, 64, 64, generator=generator)
    mask = torch.zeros(len(studies), views, dtype=torch.bool)
    for index, study in enumerate(studies):
        mask[index, : len(study.images)] = True
    view_codes = torch.zeros_like(mask, dtype=torch.long)
    for index, study in enumerate(studies):
        for slot, image in enumerate(study.images):
            view_codes[index, slot] = {"AP": 1, "PA": 2, "LATERAL": 3}.get(image.view, 0)
    orientation_codes = torch.zeros_like(mask, dtype=torch.long)
    with torch.no_grad():
        _, probability, view_logits, weights = model.forward_parts(
            pixels, mask, view_codes, orientation_codes
        )
        per_flip = {"none": view_logits.numpy()}
        for flip in flips:
            dims = {"h": [-1], "v": [-2], "hv": [-2, -1]}[flip]
            _, _, flip_logits, _ = model.forward_parts(
                torch.flip(pixels, dims=dims), mask, view_codes, orientation_codes
            )
            per_flip[flip] = flip_logits.numpy()
    return (
        [study.study_id for study in studies],
        mask.numpy(),
        view_codes.numpy(),
        orientation_codes.numpy(),
        weights.numpy(),
        per_flip,
        probability.numpy(),
    )


@pytest.mark.parametrize("pooling", POOLINGS)
def test_dump_rebuilds_the_scored_probability(pooling: str) -> None:
    config = make_config(pooling=pooling)
    torch.manual_seed(7)
    model = StudyModel(config, load_pretrained=False).eval()
    studies = fake_studies([["AP"], ["AP", "LATERAL"], ["PA", "PA", "LATERAL"]])
    flips = ("h",)

    collector = LogitCollector(("none", *flips))
    *arrays, probability = batch_parts(model, studies, flips)
    collector.add(*arrays)

    frames = attach_records(collector.frames(), studies, max_eval_views=None)
    assert frames.image_count == 6
    rebuilt = pooled_probabilities(frames, "none", pooling)
    assert np.abs(rebuilt - probability).max() < 1e-5

    # Flipped views carry their own rows, so a flip recipe can be scored later.
    assert set(frames.logits) == {"none", "h"}
    flipped = pooled_probabilities(frames, "h", pooling)
    assert flipped.shape == probability.shape
    assert np.abs(flipped - rebuilt).max() > 1e-6, "the flip must change the prediction"


@pytest.mark.parametrize("pooling", POOLINGS)
def test_per_view_logits_pool_to_the_study_logits(pooling: str) -> None:
    config = make_config(pooling=pooling)
    torch.manual_seed(11)
    model = StudyModel(config, load_pretrained=False).eval()
    pixels = torch.randn(3, 3, 3, 64, 64, generator=torch.Generator().manual_seed(3))
    mask = torch.tensor([[True, False, False], [True, True, False], [True, True, True]])
    view_codes = torch.zeros_like(mask, dtype=torch.long)
    orientation_codes = torch.zeros_like(mask, dtype=torch.long)
    with torch.no_grad():
        logits, _, view_logits, weights = model.forward_parts(
            pixels, mask, view_codes, orientation_codes
        )
    pooled = (view_logits * weights.unsqueeze(-1)).sum(dim=1)
    assert torch.allclose(pooled, logits, atol=1e-5)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)


def test_dump_survives_an_npz_round_trip() -> None:
    config = make_config()
    torch.manual_seed(5)
    model = StudyModel(config, load_pretrained=False).eval()
    studies = fake_studies([["AP"], ["AP", "LATERAL"]])
    collector = LogitCollector(("none", "h", "v"))
    *arrays, probability = batch_parts(model, studies, ("h", "v"))
    collector.add(*arrays)
    frames = attach_records(collector.frames(), studies, max_eval_views=None)

    buffer = io.BytesIO()
    write_dump(buffer, frames, {"split": "valid", "pooling": config.pooling})
    buffer.seek(0)
    reloaded, meta = read_dump(buffer)

    assert meta["split"] == "valid"
    assert reloaded.studies == frames.studies
    assert reloaded.dicom_ids == frames.dicom_ids
    assert np.array_equal(reloaded.image_study, frames.image_study)
    assert np.allclose(reloaded.weights, frames.weights, atol=1e-6)
    for flip in ("none", "h", "v"):
        assert np.allclose(reloaded.logits[flip], frames.logits[flip], atol=1e-5)
    assert np.abs(pooled_probabilities(reloaded, "none", config.pooling) - probability).max() < 1e-5


def test_attach_records_rejects_a_row_mismatch() -> None:
    config = make_config()
    torch.manual_seed(9)
    model = StudyModel(config, load_pretrained=False).eval()
    studies = fake_studies([["AP", "LATERAL"], ["PA"]])
    collector = LogitCollector(("none",))
    *arrays, _ = batch_parts(model, studies, ())
    collector.add(*arrays)
    frames = collector.frames()

    # max_eval_views=1 drops an image per multi-view study, so the dump no longer lines up.
    with pytest.raises(ValueError, match="selected"):
        attach_records(frames, studies, max_eval_views=1)

    swapped = [studies[1], studies[0]]
    with pytest.raises(ValueError, match="different order"):
        attach_records(frames, swapped, max_eval_views=None)


def test_parse_flips_rejects_nonsense() -> None:
    assert parse_flips("none,h") == ("none", "h")
    assert parse_flips(" none ,, hv ") == ("none", "hv")
    with pytest.raises(ValueError, match="must contain 'none'"):
        parse_flips("h")
    with pytest.raises(ValueError, match="unknown flips"):
        parse_flips("none,diagonal")
    with pytest.raises(ValueError, match="duplicate"):
        parse_flips("none,none")
    with pytest.raises(ValueError, match="empty"):
        parse_flips("  ")
