"""Lightning orchestration for generative RSB training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from RSB.backbone.ncsnpp.ncsnpp_utils.op.backend import PYTORCH_NATIVE_BACKEND, activate_operator_backend
from RSB.callbacks import (
    BestModelExport,
    EMACallback,
    FullStateModelCheckpoint,
    GenerativeSampleMetrics,
    ValidationEarlyStopping,
)
from RSB.data import ComplexSpecDataset
from RSB.modeling_rsb import RSB, validate_training_method
from RSB.pipelines.generative import RSBTrainingPipeline
from RSB.training.predictive import _run_path_from_checkpoint, _wandb_enabled
from RSB.training.runtime import distributed_rank, resolve_trainer_runtime
from RSB.utils.config import Config, read_config_from_yaml


@dataclass(frozen=True)
class GenerativeRun:
    """Resolved generative training run metadata."""

    config: Config
    run_path: Path
    checkpoint_path: Path | None


def prepare_generative_run(
    config: Config,
    *,
    checkpoint_path: Path | None = None,
    now: datetime | None = None,
    run_id: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> GenerativeRun:
    """Create a generative run or restore its persisted configuration."""
    if checkpoint_path is not None:
        run_path = _run_path_from_checkpoint(checkpoint_path.resolve())
        resumed_config = read_config_from_yaml(run_path / "config.yml")
        resumed_config.update(overrides or {})
        resumed_config.save(run_path)
        return GenerativeRun(resumed_config, run_path, run_path / "checkpoints" / "last.ckpt")

    timestamp = (now or datetime.now()).strftime("%m%d%H%M")
    run_name = config.get("run_name") or f"rsb_generative_{timestamp}"
    run_path = Path(config.run_dir).expanduser().resolve() / run_name
    if distributed_rank() > 0:
        child_config = read_config_from_yaml(run_path / "config.yml")
        if child_config.get("run_type") != "generative":
            raise ValueError(f"Distributed worker found a non-generative run at {run_path}")
        return GenerativeRun(child_config, run_path, None)
    run_path.mkdir(parents=True, exist_ok=False)
    config.update(
        {
            "run_type": "generative",
            "run_name": run_name,
            "run_id": run_id or (wandb.util.generate_id() if _wandb_enabled(config) else None),
            "run_path": str(run_path),
            "output_path": str(run_path),
        }
    )
    config.save(run_path)
    return GenerativeRun(config, run_path, None)


def build_generative_dataloader(config: Config, subset: str) -> DataLoader[Any]:
    """Build an RSB dataloader, including offline posterior means when required."""
    validate_training_method(config.training_method)
    requires_posterior_mean = config.training_method == "regularization"
    dataset = ComplexSpecDataset(
        config,
        dataset=config.dataset,
        subset=subset,
        shuffle_spec=subset == "train",
        return_spec=True,
        posterior_mean_from=config.get("posterior_mean_from", "NCSN++M") if requires_posterior_mean else None,
        dummy=config.get("dummy", False),
    )
    num_workers = int(config.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=subset == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_generative_callbacks(config: Config, run_path: Path) -> list[Any]:
    """Build optional EMA, export, resumable checkpoint, and early stopping callbacks."""
    callbacks: list[Any] = []
    if config.get("ema", True):
        callbacks.append(EMACallback(decay=float(config.get("ema_rate", 0.999))))
    callbacks.extend(
        [
            GenerativeSampleMetrics(
                config,
                valid_samples=int(config.get("valid_metric_samples", 50)),
                test_samples=int(config.get("test_metric_samples", 5)),
                num_steps=int(config.get("metric_num_steps", 20)),
            ),
            BestModelExport(
                run_path,
                config,
                monitor="valid/PESQ_per_epoch",
                mode="max",
                config_field="best_pesq",
            ),
            FullStateModelCheckpoint(
                run_path,
                save_state_steps=int(config.get("save_state_steps", 1000)),
                checkpoints_total_limit=int(config.get("checkpoints_total_limit", 3)),
            ),
            ValidationEarlyStopping(
                patience=int(config.get("patience", 20)),
                monitor="valid/SI_SDR_per_epoch",
                mode="max",
                config=config,
                run_path=run_path,
                best_field="best_sisdr",
            ),
        ]
    )
    return callbacks


def build_generative_pipeline(config: Config) -> RSBTrainingPipeline:
    """Instantiate RSB and its Lightning training pipeline from configuration."""
    model = RSB(
        backbone=config.generative_backbone,
        training_method=config.training_method,
        training_target=config.training_target,
        loss_weight_type=config.loss_weight_type,
        bridge_type=config.bridge_type,
        device="cpu",
    )
    return RSBTrainingPipeline(
        model,
        learning_rate=float(config.learning_rate),
        optimizer_name=config.get("optimizer", "Adam"),
        reduction=config.get("reduction", "sum"),
        regularization_weight=config.get("regularization_weight", "quadratic"),
        time_loss_weight=float(config.get("time_loss_weight", 1e-3)),
        t_min=float(config.get("t_min", 1e-4)),
        t_max=float(config.get("t_max", 1.0)),
    )


def start_generative_training(
    config: Config,
    *,
    checkpoint_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Train or resume the generative RSB model and return its run directory."""
    validate_training_method(config.training_method)
    if config.training_method != "none":
        from RSB.workflows.posterior import validate_posterior_means

        config.update({"posterior_mean_validation": validate_posterior_means(config)})
    run = prepare_generative_run(config, checkpoint_path=checkpoint_path, overrides=overrides)
    activate_operator_backend(run.config.get("ncsnpp_operator_backend", PYTORCH_NATIVE_BACKEND))
    L.seed_everything(int(run.config.get("seed", 10)), workers=True)
    if _wandb_enabled(run.config):
        logger: WandbLogger | bool = WandbLogger(
            project="RSB",
            name=run.config.run_name,
            id=run.config.run_id,
            save_dir=str(run.run_path),
            resume="allow" if run.checkpoint_path is not None else None,
        )
        logger.log_hyperparams(run.config.dict())
    else:
        logger = False

    runtime = resolve_trainer_runtime(run.config)
    trainer = L.Trainer(
        accelerator=runtime.accelerator,
        devices=runtime.devices,
        strategy=runtime.strategy,
        num_nodes=int(run.config.get("num_nodes", 1)),
        precision=runtime.precision,
        max_epochs=int(run.config.num_epoch),
        default_root_dir=run.run_path,
        logger=logger,
        callbacks=build_generative_callbacks(run.config, run.run_path),
        log_every_n_steps=int(run.config.get("log_steps", 10)),
    )
    trainer.fit(
        build_generative_pipeline(run.config),
        train_dataloaders=build_generative_dataloader(run.config, "train"),
        val_dataloaders=build_generative_dataloader(run.config, "valid"),
        ckpt_path=str(run.checkpoint_path) if run.checkpoint_path is not None else None,
    )
    return run.run_path
