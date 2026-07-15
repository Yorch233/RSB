"""Orchestration for predictive-model training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from RSB.backbone.ncsnpp.ncsnpp_utils.op.backend import PYTORCH_NATIVE_BACKEND, activate_operator_backend
from RSB.callbacks import BestModelExport, EMACallback, FullStateModelCheckpoint, ValidationEarlyStopping
from RSB.data import ComplexSpecDataset
from RSB.pipelines.predictive import PredictiveModelTrainingPipeline
from RSB.training.runtime import distributed_rank, resolve_trainer_runtime
from RSB.utils.config import Config, read_config_from_yaml

if TYPE_CHECKING:
    from torch import nn

METHOD_BACKBONES = {"NCSN++M": "ncsnpp_base"}


def _wandb_enabled(config: Config) -> bool:
    """Return whether the configured experiment logger is WandB."""
    return config.get("logger", "wandb") == "wandb"


@dataclass(frozen=True)
class PredictiveRun:
    """Resolved training run metadata."""

    config: Config
    run_path: Path
    checkpoint_path: Path | None


def _run_path_from_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.is_dir():
        if (checkpoint_path / "checkpoints" / "last.ckpt").is_file():
            return checkpoint_path
        if checkpoint_path.name == "checkpoints" and (checkpoint_path / "last.ckpt").is_file():
            return checkpoint_path.parent
        raise ValueError(f"No checkpoints/last.ckpt found under {checkpoint_path}")
    if checkpoint_path.name != "last.ckpt" or checkpoint_path.parent.name != "checkpoints":
        raise ValueError("A resume checkpoint must be a run directory, checkpoints directory, or checkpoints/last.ckpt")
    return checkpoint_path.parent.parent


def prepare_predictive_run(
    config: Config,
    *,
    method: str,
    checkpoint_path: Path | None = None,
    now: datetime | None = None,
    run_id: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> PredictiveRun:
    """Create a new run or resolve an existing run for resume."""
    if method not in METHOD_BACKBONES:
        supported = ", ".join(METHOD_BACKBONES)
        raise ValueError(f"Unsupported predictive method {method!r}; choose from {supported}")

    if checkpoint_path is not None:
        run_path = _run_path_from_checkpoint(checkpoint_path.resolve())
        resumed_config = read_config_from_yaml(run_path / "config.yml")
        if resumed_config.get("predictive_method") != method:
            raise ValueError(
                f"Checkpoint method {resumed_config.get('predictive_method')!r} does not match requested {method!r}"
            )
        resumed_config.update(overrides or {})
        resumed_config.save(run_path)
        return PredictiveRun(resumed_config, run_path, run_path / "checkpoints" / "last.ckpt")

    timestamp = (now or datetime.now()).strftime("%m%d%H%M")
    run_name = config.get("run_name") or f"rsb_predictive_{timestamp}"
    run_path = Path(config.run_dir).expanduser().resolve() / run_name
    if distributed_rank() > 0:
        child_config = read_config_from_yaml(run_path / "config.yml")
        if child_config.get("run_type") != "predictive":
            raise ValueError(f"Distributed worker found a non-predictive run at {run_path}")
        if child_config.get("predictive_method") != method:
            raise ValueError(
                f"Distributed worker method {child_config.get('predictive_method')!r} "
                f"does not match requested {method!r}"
            )
        return PredictiveRun(child_config, run_path, None)
    run_path.mkdir(parents=True, exist_ok=False)
    config.update(
        {
            "predictive_method": method,
            "run_type": "predictive",
            "predictive_backbone": METHOD_BACKBONES[method],
            "run_name": run_name,
            "run_id": run_id or (wandb.util.generate_id() if _wandb_enabled(config) else None),
            "run_path": str(run_path),
            "output_path": str(run_path),
        }
    )
    config.save(run_path)
    return PredictiveRun(config, run_path, None)


def build_predictive_model(config: Config) -> nn.Module:
    """Instantiate the predictive backbone selected by the training method."""
    from RSB.backbone import BackboneRegister

    return BackboneRegister.fetch(config.predictive_backbone)(discriminative=True)


def build_predictive_dataloader(config: Config, subset: str) -> DataLoader[Any]:
    """Build a clean/noisy complex-spectrum dataloader."""
    dataset = ComplexSpecDataset(
        config,
        dataset=config.dataset,
        subset=subset,
        shuffle_spec=subset == "train",
        return_spec=True,
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


def build_predictive_callbacks(config: Config, run_path: Path) -> list[Any]:
    """Build full-state checkpointing, best-weight export, and early stopping."""
    callbacks: list[Any] = []
    if config.get("ema", True):
        callbacks.append(EMACallback(decay=float(config.get("ema_rate", 0.999))))
    callbacks.extend(
        [
            BestModelExport(run_path, config),
            FullStateModelCheckpoint(
                run_path,
                save_state_steps=int(config.get("save_state_steps", 1000)),
                checkpoints_total_limit=int(config.get("checkpoints_total_limit", 3)),
            ),
            ValidationEarlyStopping(patience=int(config.get("predictive_patience", 50))),
        ]
    )
    return callbacks


def start_predictive_training(
    config: Config,
    *,
    method: str,
    checkpoint_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Train or resume a predictive model and return its run directory."""
    run = prepare_predictive_run(
        config,
        method=method,
        checkpoint_path=checkpoint_path,
        overrides=overrides,
    )
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
    pipeline = PredictiveModelTrainingPipeline(
        build_predictive_model(run.config),
        method=method,
        learning_rate=float(run.config.learning_rate),
        optimizer_name=run.config.get("optimizer", "Adam"),
        reduction=run.config.get("reduction", "sum"),
    )
    trainer = L.Trainer(
        accelerator=runtime.accelerator,
        devices=runtime.devices,
        strategy=runtime.strategy,
        num_nodes=int(run.config.get("num_nodes", 1)),
        precision=runtime.precision,
        max_epochs=int(run.config.num_epoch),
        default_root_dir=run.run_path,
        logger=logger,
        callbacks=build_predictive_callbacks(run.config, run.run_path),
        log_every_n_steps=int(run.config.get("log_steps", 10)),
    )
    trainer.fit(
        pipeline,
        train_dataloaders=build_predictive_dataloader(run.config, "train"),
        val_dataloaders=build_predictive_dataloader(run.config, "valid"),
        ckpt_path=str(run.checkpoint_path) if run.checkpoint_path is not None else None,
    )
    return run.run_path
