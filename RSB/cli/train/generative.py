"""Generative RSB training command."""

from enum import StrEnum
from pathlib import Path

import typer

from RSB.cli.train import common
from RSB.cli.train.common import Logger, Optimizer


class Schedule(StrEnum):
    """Supported Schrodinger Bridge schedules."""

    VP = "VP"
    VE = "VE"


class TrainingMethod(StrEnum):
    """Supported generative training methods exposed by the CLI."""

    NONE = "none"
    REGULARIZATION = "regularization"


class TrainingTarget(StrEnum):
    """Supported SDE network targets."""

    DATA = "data"
    NOISE = "noise"
    SCORE = "score"
    VECTOR = "vector"


class RegularizationWeight(StrEnum):
    """Supported time-dependent regularization weights."""

    QUADRATIC = "quadratic"
    COSINE = "cosine"
    LINEAR = "linear"


def generative(
    config_path: Path = typer.Option(
        common.DEFAULT_CONFIG_PATH,
        "--config",
        dir_okay=False,
        readable=True,
        help="Run configuration YAML; defaults to config/default.yml.",
    ),
    max_epoch: int | None = typer.Option(None, "--max-epoch", "--max_epoch", min=1),
    learning_rate: float | None = typer.Option(None, "--learning-rate", "--learning_rate", min=0.0),
    dataset: str | None = typer.Option(None, help="Registered dataset ID to use for training."),
    run_name: str | None = typer.Option(None, "--run-name", "--run_name"),
    run_dir: Path | None = typer.Option(None, "--run-dir", "--run_dir", file_okay=False),
    batch_size: int | None = typer.Option(None, "--batch-size", "--batch_size", min=1),
    optimizer: Optimizer | None = typer.Option(None, case_sensitive=True),
    logger: Logger | None = typer.Option(None, case_sensitive=False),
    ema: bool | None = typer.Option(None, "--ema/--no-ema"),
    ema_rate: float | None = typer.Option(None, "--ema-rate", "--ema_rate", min=0.0, max=0.999999),
    patience: int | None = typer.Option(None, min=1, help="Validation-loss early-stopping patience."),
    num_workers: int | None = typer.Option(None, "--num-workers", "--num_workers", min=0),
    seed: int | None = typer.Option(None),
    precision: str | None = typer.Option(None, help="Lightning precision, for example 32-true or bf16-mixed."),
    accelerator: str | None = typer.Option(None, help="Lightning accelerator."),
    devices: str | None = typer.Option(None, help="Lightning device selection."),
    strategy: str | None = typer.Option(None, help="Lightning distributed strategy."),
    num_nodes: int | None = typer.Option(None, "--num-nodes", "--num_nodes", min=1),
    log_steps: int | None = typer.Option(None, "--log-steps", "--log_steps", min=1),
    schedule: Schedule = typer.Option(Schedule.VE, case_sensitive=True, help="VP or VE."),
    training_method: TrainingMethod = typer.Option(
        TrainingMethod.REGULARIZATION,
        "--training-method",
        "--training_method",
        case_sensitive=True,
        help="Choose none for vanilla SB or regularization for RSB.",
    ),
    training_target: TrainingTarget | None = typer.Option(
        None,
        "--training-target",
        "--training_target",
        case_sensitive=True,
    ),
    posterior_mean_from: str | None = typer.Option(
        None,
        "--posterior-mean-from",
        "--posterior_mean_from",
        help="Offline posterior-mean directory name; defaults to NCSN++M unless training-method is none.",
    ),
    regularization_weight: RegularizationWeight = typer.Option(
        RegularizationWeight.QUADRATIC,
        "--regularization-weight",
        "--regularization_weight",
        case_sensitive=True,
        help="Time-dependent regularization weight.",
    ),
    generative_backbone: str | None = typer.Option(None, "--generative-backbone", "--generative_backbone"),
    loss_weight_type: str | None = typer.Option(None, "--loss-weight-type", "--loss_weight_type"),
    reduction: str | None = typer.Option(None, help="Loss reduction: mean or sum."),
    time_loss_weight: float | None = typer.Option(None, "--time-loss-weight", "--time_loss_weight", min=0.0),
    t_min: float | None = typer.Option(None, "--t-min", "--t_min", min=0.0),
    t_max: float | None = typer.Option(None, "--t-max", "--t_max", min=0.0),
    resume: bool = typer.Option(False),
    checkpoint_path: Path | None = typer.Option(None, "--checkpoint-path", "--checkpoint_path", exists=True),
    yes: bool = typer.Option(False, "--yes", help="Skip operator fallback and training confirmations."),
) -> None:
    """Train the generative RSB model."""
    from RSB.training.generative import start_generative_training

    common._validate_resume(resume, checkpoint_path)
    if training_method is TrainingMethod.NONE and posterior_mean_from is not None:
        raise typer.BadParameter("--posterior-mean-from is not applicable when --training-method is none")
    resolved_posterior_mean = None if training_method is TrainingMethod.NONE else posterior_mean_from or "NCSN++M"
    config, overrides = common._load_config(
        config_path,
        use_local_runtime_defaults=True,
        num_epoch=max_epoch,
        learning_rate=learning_rate,
        dataset=dataset,
        run_name=run_name,
        run_dir=run_dir,
        batch_size=batch_size,
        optimizer=optimizer,
        logger=logger,
        ema=ema,
        ema_rate=ema_rate,
        patience=patience,
        num_workers=num_workers,
        seed=seed,
        precision=precision,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        num_nodes=num_nodes,
        log_steps=log_steps,
        bridge_type=schedule,
        training_method=training_method,
        training_target=training_target,
        posterior_mean_from=resolved_posterior_mean,
        regularization_weight=regularization_weight,
        generative_backbone=generative_backbone,
        loss_weight_type=loss_weight_type,
        reduction=reduction,
        time_loss_weight=time_loss_weight,
        t_min=t_min,
        t_max=t_max,
    )
    if resolved_posterior_mean is None:
        config.update({"posterior_mean_from": None})
        overrides["posterior_mean_from"] = None
    if float(config.t_min) >= float(config.t_max):
        raise typer.BadParameter("--t-min must be smaller than --t-max")
    config, overrides = common._prepare_training_command(
        config,
        overrides,
        checkpoint_path=checkpoint_path,
        assume_yes=yes,
    )
    start_generative_training(config, checkpoint_path=checkpoint_path, overrides=overrides)
