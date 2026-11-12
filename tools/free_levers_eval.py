"""Night-1 free single-model levers: A2 soup + A3 multi-scale TTA (CPU).

Runs on local CPU (torch 2.9 cpu). Loads p8 checkpoints at arbitrary image_size
by bypassing the load_checkpoint image_size equality check (weights are size-agnostic
for efficientnet_b0 + image_mean pooling).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cxr.config import Config  # noqa: E402
from cxr.metrics import summarize  # noqa: E402
from cxr.model import StudyModel  # noqa: E402
from cxr.records import read_available  # noqa: E402
from cxr.runtime import device_for, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPTS = {
    "b0_224": ROOT / "runs/p8/p8_b0_224_12ep_s42",
    "b0_320": ROOT / "runs/p8/p8_b0_320_12ep_s42",
    "b0_448": ROOT / "runs/p8/p8_b0_448_12ep_s42",
}
NATIVE = {"b0_224": 224, "b0_320": 320, "b0_448": 448}
SCALES = (224, 320, 448)

base = json.loads((CKPTS["b0_224"] / "config.json").read_text(encoding="utf-8"))
DATA_ROOT = Path(base["data_root"])


def make_cfg(image_size: int, name: str) -> Config:
    cfg = dict(base)
    cfg.update(
        name=name,
        image_size=image_size,
        device="cpu",
        amp=False,
        num_workers=0,
        cpu_threads=8,
        output_dir=str(ROOT / "_work" / "night1" / name),
    )
    for key in ("data_root", "output_dir", "init_from"):
        if isinstance(cfg.get(key), str):
            cfg[key] = Path(cfg[key])
    return Config.model_validate(cfg)


def load_any_size(ckpt_dir: Path, cfg: Config, device: torch.device) -> StudyModel:
    state = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=True)
    saved = state["config"]
    for key in ("backbone", "pooling", "loss", "view_conditioning", "orientation_conditioning"):
        sv = saved.get(key)
        if key in ("view_conditioning", "orientation_conditioning") and sv is None:
            sv = False
        if sv != getattr(cfg, key):
            raise ValueError(f"{key} mismatch saved={sv} cfg={getattr(cfg, key)}")
    model = StudyModel(cfg, load_pretrained=False)
    model.load_state_dict(state["model"])  # strict=True -> fails loudly if arch differs
    return model.to(device)


def load_sd(ckpt_dir: Path, which: str) -> dict[str, torch.Tensor]:
    st = torch.load(ckpt_dir / f"{which}.pt", map_location="cpu", weights_only=True)
    return st["model"]


def prob2logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def logit2prob(l: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(l, -50, 50)))


data = read_available(DATA_ROOT, splits=("valid",))
studies = sorted(data["valid"].studies, key=lambda s: s.study_id)
truth = np.zeros((len(studies), 10), dtype=np.float32)
for r, s in enumerate(studies):
    for lab in (s.labels or ()):
        truth[r, lab - 1] = 1.0


def auc(p: np.ndarray) -> float:
    return float(summarize(truth, p, 0.0)["macro_auc"])


def eval_native(ckpt_dir: Path, cfg: Config) -> np.ndarray:
    device = device_for(cfg)
    model = load_any_size(ckpt_dir, cfg, device)
    _, p = score(model, studies, cfg, device)
    return p


print("== A3: multi-scale TTA (each checkpoint @ 224/320/448) ==", flush=True)
probs: dict[tuple[str, int], np.ndarray] = {}
for cname, cdir in CKPTS.items():
    for scale in SCALES:
        cfg = make_cfg(scale, f"{cname}_{scale}")
        _, p = score(load_any_size(cdir, cfg, device_for(cfg)), studies, cfg, device_for(cfg))
        probs[(cname, scale)] = p
        print(f"  {cname}@{scale}: macro_auc={auc(p):.4f}", flush=True)

print("\n== A3 results ==", flush=True)
for cname in CKPTS:
    nat = probs[(cname, NATIVE[cname])]
    multi = np.mean([probs[(cname, s)] for s in SCALES], axis=0)
    print(f"  {cname}: native={auc(nat):.4f} multiscale_prob={auc(multi):.4f}", flush=True)

all_prob = np.mean([probs[(c, s)] for c in CKPTS for s in SCALES], axis=0)
all_logit = logit2prob(np.mean([prob2logit(probs[(c, s)]) for c in CKPTS for s in SCALES], axis=0))
print(f"  FULL-9 prob-avg = {auc(all_prob):.4f}", flush=True)
print(f"  FULL-9 logit-avg = {auc(all_logit):.4f}", flush=True)

print("\n== A2: best+last soup (native scale) ==", flush=True)
for cname, cdir in CKPTS.items():
    scale = NATIVE[cname]
    cfg = make_cfg(scale, f"soup_bl_{cname}")
    device = device_for(cfg)
    best_sd = load_sd(cdir, "best")
    last_sd = load_sd(cdir, "last")
    soup = {k: (best_sd[k].float() + last_sd[k].float()) / 2 for k in best_sd}
    model = StudyModel(cfg, load_pretrained=False)
    model.load_state_dict(soup)
    _, p = score(model.to(device), studies, cfg, device)
    print(f"  {cname} best+last soup = {auc(p):.4f}", flush=True)

print("\n== A2: cross-resolution soup (avg 3 best, eval at each scale) ==", flush=True)
best_sds = [load_sd(cdir, "best") for cdir in CKPTS.values()]
soup3 = {k: (best_sds[0][k].float() + best_sds[1][k].float() + best_sds[2][k].float()) / 3
         for k in best_sds[0]}
for scale in SCALES:
    cfg = make_cfg(scale, f"soup3_{scale}")
    device = device_for(cfg)
    model = StudyModel(cfg, load_pretrained=False)
    model.load_state_dict(soup3)
    _, p = score(model.to(device), studies, cfg, device)
    print(f"  soup3@{scale} = {auc(p):.4f}", flush=True)

print("\nDONE", flush=True)
