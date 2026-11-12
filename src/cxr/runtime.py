"""Training, checkpointing, deterministic evaluation, and optional cache verification."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic_core import PydanticUndefined
from torch.utils.data import DataLoader, WeightedRandomSampler

from .artifacts import (
    load_checkpoint,
    make_checkpoint,
    write_checkpoint,
    write_environment,
    write_json,
)
from .commits import reconcile_run, snapshot_model
from .config import Config
from .data import (
    StudyDataset,
    cache_root_from_env,
    cache_size_from_env,
    collate_studies,
)
from .holdout import Holdout
from .logits import FLIP_DIMS, LogitCollector
from .metrics import Metrics, summarize
from .model import StudyModel
from .records import StudyRecord, read_available, study_view_group
from .splits import choose_studies, manifest_studies

# These change execution but not the saved weights or optimizer semantics.
RESUME_IGNORED = {
    "epochs",
    "data_root",
    "num_workers",
    "cpu_threads",
    "log_interval",
    "channels_last",
    "cudnn_benchmark",
    "tta",
    "max_eval_views",
    "reweight_views",
}


def device_for(config: Config) -> torch.device:
    torch.set_num_threads(config.cpu_threads)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
    return device


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init(_worker_id: int) -> None:
    # Four workers inheriting four intra-op threads oversubscribe an 8-core host.
    torch.set_num_threads(1)


def _loader(
    studies: Sequence[StudyRecord],
    config: Config,
    training: bool,
    epoch: int = 0,
    max_eval_views: int | None = None,
) -> DataLoader[Any]:
    dataset = StudyDataset(
        studies,
        config.image_size,
        training,
        config.max_train_views,
        config.seed,
        cache_root=cache_root_from_env(),
        cache_size=cache_size_from_env(config.cache_size),
        max_eval_views=None if training else max_eval_views,
        data_root=config.data_root,
        draft=config.draft,
        clahe=config.clahe,
        aug_level=config.aug_level,
    )
    dataset.epoch = epoch
    generator = torch.Generator().manual_seed(config.seed + epoch)
    workers = config.num_workers
    options: dict[str, Any] = {}
    if workers > 0:
        options = {
            "persistent_workers": True,
            "prefetch_factor": 4,
            "worker_init_fn": _worker_init,
        }
    sampler = None
    if training and config.patient_balanced:
        # 1 / (studies of this patient) so every patient carries the same total weight.
        counts: dict[str, int] = {}
        for study in studies:
            counts[study.subject_id] = counts.get(study.subject_id, 0) + 1
        weights = [1.0 / counts[study.subject_id] for study in studies]
        sampler = WeightedRandomSampler(
            weights, num_samples=len(studies), replacement=True, generator=generator
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_studies,
        generator=generator,
        pin_memory=config.pin_memory and config.device == "cuda",
        **options,
    )


def _cache_info(requested_size: int | None = None) -> dict[str, object] | None:
    root = cache_root_from_env()
    if root is None:
        return None
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"CXR_CACHE_ROOT={root} has no manifest.json; run tools/build_cache.py first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("cache_id"):
        raise ValueError(f"Cache manifest lacks cache_id: {manifest_path}")
    if manifest.get("partial"):
        raise ValueError(f"Refusing a partial cache: {manifest_path}")
    size = int(manifest.get("size", 0))
    if requested_size is not None and size != requested_size:
        raise ValueError(f"CXR_CACHE_SIZE={requested_size} differs from cache size {size}")
    return {
        "root": str(root.resolve()),
        "cache_id": str(manifest["cache_id"]),
        "size": size,
        "images": int(manifest.get("images", 0)),
    }


def _scheduler(optimizer: torch.optim.Optimizer, config: Config) -> Any:
    horizon = config.schedule_epochs or config.epochs
    warmup = min(config.warmup_epochs, max(0, horizon - 1))
    if warmup > 0:
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, horizon - warmup)
                ),
            ],
            milestones=[warmup],
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=horizon)


def _config_mismatches(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for key, value in current.items():
        if key in RESUME_IGNORED:
            continue
        if key in saved:
            if saved[key] != value:
                mismatches.append(key)
            continue
        default = Config.model_fields[key].get_default(call_default_factory=True)
        if default is PydanticUndefined or value != default:
            mismatches.append(key)
    return mismatches


def _capture_into(store: dict[str, Any]) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """Sink for ``StudyModel.study_loss`` that stashes the per-view tensors."""

    def sink(view_logits: torch.Tensor, weights: torch.Tensor) -> None:
        store["view_logits"] = view_logits.detach()
        store["weights"] = weights.detach()

    return sink


@torch.no_grad()
def score(
    model: StudyModel,
    studies: Sequence[StudyRecord],
    config: Config,
    device: torch.device,
    max_batches: int | None = None,
    view_weights: dict[str, float] | None = None,
    collector: LogitCollector | None = None,
) -> tuple[Metrics, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    seen = 0
    for index, (
        pixels,
        mask,
        view_codes,
        orientation_codes,
        labels,
        batch_studies,
        _,
    ) in enumerate(_loader(studies, config, False, max_eval_views=config.max_eval_views)):
        if max_batches is not None and index >= max_batches:
            break
        if collector is not None:
            # Keep CPU copies of the identifiers before the tensors move to the device.
            mask_rows = np.array(mask, copy=True)
            view_rows = np.array(view_codes, copy=True)
            orientation_rows = np.array(orientation_codes, copy=True)
        pixels, mask, view_codes, orientation_codes, labels = (
            pixels.to(device, non_blocking=True),
            mask.to(device, non_blocking=True),
            view_codes.to(device, non_blocking=True),
            orientation_codes.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        captured: dict[str, Any] = {}
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda" and config.amp
        ):
            if collector is None:
                loss, probability = model.study_loss(
                    pixels, mask, view_codes, orientation_codes, labels, config.loss
                )
            else:
                loss, probability = model.study_loss(
                    pixels, mask, view_codes, orientation_codes, labels, config.loss,
                    sink=_capture_into(captured),
                )
                for flip in collector.extra:
                    flipped = torch.flip(pixels, dims=FLIP_DIMS[flip])
                    _, _, flip_logits, _ = model.forward_parts(
                        flipped, mask, view_codes, orientation_codes
                    )
                    captured[f"flip::{flip}"] = flip_logits.detach()
        if not torch.isfinite(loss) or not torch.isfinite(probability).all():
            raise ValueError("Nonfinite validation output")
        if collector is not None:
            per_flip = {"none": captured["view_logits"].float().cpu().numpy()}
            for flip in collector.extra:
                per_flip[flip] = captured[f"flip::{flip}"].float().cpu().numpy()
            collector.add(
                list(batch_studies),
                mask_rows,
                view_rows,
                orientation_rows,
                captured["weights"].float().cpu().numpy(),
                per_flip,
            )
        size = labels.shape[0]
        losses.append(float(loss.item()) * size)
        seen += size
        scores.append(probability.float().cpu().numpy())
        targets.append(labels.cpu().numpy())
    if not scores:
        raise ValueError("No validation batches")
    probabilities = np.concatenate(scores)
    truth = np.concatenate(targets)
    groups = [study_view_group(study) for study in studies][: len(probabilities)]
    return (
        summarize(truth, probabilities, sum(losses) / seen, groups, view_weights),
        probabilities,
    )


def train(config: Config, resume: Path | None = None, holdout: Holdout | None = None) -> None:
    config.require_supervised_selection()
    device = device_for(config)
    seed_all(config.seed)
    data = read_available(config.data_root, splits=("train", "valid"))
    if "train" not in data:
        raise ValueError("train.csv is required")
    run = config.output_dir
    run_files = (
        "config.json",
        "environment.json",
        "split_manifest.json",
        "metrics.jsonl",
        "best.pt",
        "last.pt",
        "best_metrics.json",
        "validation_predictions.csv",
        "cache_manifest.json",
    )
    if resume is None and any((run / filename).exists() for filename in run_files):
        raise FileExistsError(f"Run artifacts already exist in {run}; use --resume")
    if resume is not None and resume.resolve() != (run / "last.pt").resolve():
        raise ValueError("Resume must use this run's last.pt")
    run.mkdir(parents=True, exist_ok=True)
    if resume is None:
        from .codesnapshot import write_code_snapshot

        snapshot = write_code_snapshot(run, config, tag="train")
        print(f"[snapshot] code_snapshot sha256={snapshot['archive']['sha256'][:12]}", flush=True)
    cache = _cache_info(cache_size_from_env(config.cache_size))
    if resume is None:
        train_studies, valid_studies, manifest = choose_studies(config, data, holdout)
        model = StudyModel(config).to(device)
        if config.channels_last:
            model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
        if config.init_from is not None:
            if not config.init_from.is_file():
                raise FileNotFoundError(f"init_from checkpoint missing: {config.init_from}")
            state = torch.load(config.init_from, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            print(f"[init] weights from {config.init_from} (fresh optimiser, epoch 1)", flush=True)
        write_json(run / "split_manifest.json", manifest)
        write_json(run / "config.json", config.model_dump(mode="json"))
        write_environment(run / "environment.json", config, device)
        if cache is not None:
            write_json(run / "cache_manifest.json", cache)
        start_epoch = 1
        best_metrics: Metrics | None = None
        best_model: dict[str, torch.Tensor] | None = None
        best_probabilities: torch.Tensor | None = None
        best_validation_study_ids: list[str] | None = None
    else:
        manifest = json.loads((run / "split_manifest.json").read_text(encoding="utf-8"))
        if manifest["csv_sha256"] != {name: part.sha256 for name, part in data.items()}:
            raise ValueError("CSV hashes differ from saved split manifest")
        train_map = {study.study_id: study for part in data.values() for study in part.studies}
        train_studies = [train_map[key] for key in manifest["train_study_ids"]]
        valid_studies = manifest_studies(data, manifest)
        saved_cache_path = run / "cache_manifest.json"
        saved_cache = (
            json.loads(saved_cache_path.read_text(encoding="utf-8"))
            if saved_cache_path.is_file()
            else None
        )
        if (saved_cache is None) != (cache is None):
            raise ValueError("Cache configuration differs from the saved run")
        if saved_cache is not None and cache is not None:
            if saved_cache.get("cache_id") != cache.get("cache_id") or saved_cache.get(
                "size"
            ) != cache.get("size"):
                raise ValueError("Cache content differs from the saved run")
        model, state = load_checkpoint(resume, config, device)
        if state["config"].get("reweight_views", False):
            raise ValueError("Cannot resume a historical reweight_views run; start a fresh run")
        if config.channels_last:
            model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
        saved_config = state["config"]
        current_config = config.model_dump(mode="json")
        mismatches = _config_mismatches(saved_config, current_config)
        if mismatches:
            raise ValueError(
                f"Resume config differs in {mismatches}; only runtime options may change"
            )
        if config.epochs < int(state["epoch"]):
            raise ValueError("Requested epochs precede checkpoint epoch")
        reconcile_run(run, state, valid_studies)
        if config.epochs != int(saved_config["epochs"]):
            with (run / "resume_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "from_epoch": int(state["epoch"]),
                            "previous_target_epochs": int(saved_config["epochs"]),
                            "new_target_epochs": config.epochs,
                            "scheduler_policy": (
                                "restart cosine decay at configured learning_rate "
                                "over remaining epochs; warmup is not reapplied"
                            ),
                        }
                    )
                    + "\n"
                )
            write_json(run / "config.json", current_config)
        start_epoch = int(state["epoch"]) + 1
        best_metrics = state["best_metrics"]
        best_model = state.get("best_model")
        best_probabilities = state.get("best_probabilities")
        best_validation_study_ids = state.get("best_validation_study_ids")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.amp)
    if resume is None:
        initial = make_checkpoint(model, optimizer, scheduler, scaler, 0, None, config)
        initial.update(
            epoch_metrics=None,
            best_model=None,
            best_probabilities=None,
            best_validation_study_ids=None,
        )
        write_checkpoint(run / "last.pt", initial)
        reconcile_run(run, initial, valid_studies)
    else:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if config.epochs > int(state["config"]["epochs"]):
            # Extension policy: restart cosine decay at configured LR for remaining epochs.
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate
                group["initial_lr"] = config.learning_rate
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, config.epochs - int(state["epoch"]))
            )
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        numpy_state = state["numpy_rng"]
        np.random.set_state(
            (
                numpy_state[0],
                np.asarray(numpy_state[1], dtype=np.uint32),
                numpy_state[2],
                numpy_state[3],
                numpy_state[4],
            )
        )
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    for epoch in range(start_epoch, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        train_loss = torch.zeros((), device=device)
        window_loss = torch.zeros((), device=device)
        steps = 0
        window_steps = 0
        for index, (pixels, mask, view_codes, orientation_codes, labels, _, _) in enumerate(
            _loader(train_studies, config, True, epoch)
        ):
            if config.max_train_batches is not None and index >= config.max_train_batches:
                break
            pixels, mask, view_codes, orientation_codes, labels = (
                pixels.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                view_codes.to(device, non_blocking=True),
                orientation_codes.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda" and config.amp,
            ):
                loss, _ = model.study_loss(
                    pixels, mask, view_codes, orientation_codes, labels, config.loss
                )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.detach().float()
            window_loss += loss.detach().float()
            steps += 1
            window_steps += 1
            if steps == 1 or steps % config.log_interval == 0:
                # One micro-batch is a single noisy sample (batch 4 x ten rare labels, ASL
                # compresses the negatives), so print the window mean next to it.
                average = float(window_loss / window_steps)
                print(
                    f"epoch={epoch} step={steps} loss={float(loss.detach()):.6f} "
                    f"avg={average:.6f} pos={int(labels.sum())}",
                    flush=True,
                )
                window_loss.zero_()
                window_steps = 0
        if not steps:
            raise ValueError("No training batches")
        scheduler.step()
        if config.validation == "none":
            # No checkpoint selection or validation feedback in frozen final training.
            metrics: Metrics = {
                "val_loss": 0.0, "macro_auc": None, "macro_auc_defined": None,
                "defined_auc_count": 0, "macro_auc_reweighted": None,
                "macro_auc_reweighted_defined": None, "defined_auc_count_reweighted": 0,
                "macro_f1": 0.0, "per_class_auc": [None] * 10,
                "per_class_auc_reweighted": [None] * 10, "view_group_auc": {},
                "view_weights": None, "positive_counts": [0] * 10, "study_count": 0,
                "scope": "no_validation_fixed_epochs",
            }
            probabilities = np.empty((0, 10))
        else:
            metrics, probabilities = score(
                model, valid_studies, config, device, config.max_val_batches
            )
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(train_loss / steps)
        metrics["steps"] = steps
        metrics["elapsed_seconds"] = time.perf_counter() - started
        metrics["partial_smoke"] = config.partial
        selection_key = "macro_auc"
        metrics["selection_metric"] = selection_key
        old_auc = None if best_metrics is None else best_metrics["macro_auc"]
        new_auc = metrics["macro_auc"]
        better = (
            best_metrics is None
            or (new_auc is not None and (old_auc is None or new_auc > old_auc))
            or (
                new_auc is None
                and old_auc is None
                and metrics["val_loss"] < best_metrics["val_loss"]
            )
        )
        if better and config.validation != "none":
            best_metrics = metrics
            best_model = snapshot_model(model)
            best_probabilities = torch.from_numpy(probabilities.copy())
            best_validation_study_ids = [
                study.study_id for study in valid_studies[: len(probabilities)]
            ]
        committed = make_checkpoint(
            model, optimizer, scheduler, scaler, epoch, best_metrics, config
        )
        committed.update(
            epoch_metrics=metrics,
            best_model=best_model,
            best_probabilities=best_probabilities,
            best_validation_study_ids=best_validation_study_ids,
        )
        write_checkpoint(run / "last.pt", committed)
        reconcile_run(run, committed, valid_studies)
        print(
            f"epoch={epoch} val_loss={metrics['val_loss']:.6f} "
            f"macro_auc={metrics['macro_auc']} "
            f"macro_auc_reweighted={metrics['macro_auc_reweighted']} "
            f"wall_seconds={metrics['elapsed_seconds']:.1f}",
            flush=True,
        )
        if best_metrics is not None and epoch - int(best_metrics["epoch"]) >= config.patience:
            break
    if config.validation == "none":
        write_checkpoint(run / "final.pt", {
            "model": snapshot_model(model), "epoch": config.epochs,
            "best_metrics": None, "config": config.model_dump(mode="json"),
        })
