"""Predictive-model training command."""

from enum import StrEnum
from pathlib import Path

import typer

from RSB.cli.train import common
from RSB.cli.train.common import Logger, Optimizer


class PredictiveMethod(StrEnum):
    """Supported predictive-model methods."""

    NCSNPP_M = "NCSN++M"


def predictive(
    method: PredictiveMethod = typer.Option(PredictiveMethod.NCSNPP_M, case_sensitive=True),
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
    resume: bool = typer.Option(False),
    checkpoint_path: Path | None = typer.Option(None, "--checkpoint-path", "--checkpoint_path", exists=True),
    yes: bool = typer.Option(False, "--yes", help="Skip operator fallback and training confirmations."),
) -> None:
    """Train a predictive model."""
    from RSB.training.predictive import start_predictive_training

    common._validate_resume(resume, checkpoint_path)
    config, overrides = common._load_config(
        common.USER_CONFIG_PATH,
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
        predictive_patience=patience,
        num_workers=num_workers,
        seed=seed,
        precision=precision,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        num_nodes=num_nodes,
        log_steps=log_steps,
    )
    config, overrides = common._prepare_training_command(
        config,
        overrides,
        checkpoint_path=checkpoint_path,
        assume_yes=yes,
    )
    start_predictive_training(
        config,
        method=method.value,
        checkpoint_path=checkpoint_path,
        overrides=overrides,
    )
