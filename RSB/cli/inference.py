"""Predictive and generative inference command groups."""

from enum import StrEnum

import typer

from RSB.workflows.inference import run_generative_inference, run_predictive_inference

app = typer.Typer(help="Run test-set inference with trained predictive or RSB models.")


class DatasetSplit(StrEnum):
    """Supported registered dataset splits."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class Sampler(StrEnum):
    """Supported RSB sampling solvers."""

    SDE = "SDE"
    ODE = "ODE"


class SkipType(StrEnum):
    """Supported sampling timestep schedules."""

    UNIFORM = "time_uniform"
    QUADRATIC = "time_quadratic"


@app.command()
def predictive(
    run: str = typer.Option(..., help="Predictive run name or directory."),
    dataset: str | None = typer.Option(None, help="Registered dataset ID; defaults to the selected ID."),
    split: DatasetSplit = typer.Option(DatasetSplit.TEST, case_sensitive=False),
    device: str = typer.Option("auto", help="Torch device, CUDA index, or auto."),
    num_workers: int = typer.Option(0, "--num-workers", min=0),
    overwrite: bool = typer.Option(False, help="Replace conflicting or existing result WAV files."),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """Enhance a registered dataset split with a predictive model."""
    try:
        output = run_predictive_inference(
            run,
            dataset_id=dataset,
            split=split.value,
            device=device,
            num_workers=num_workers,
            overwrite=overwrite,
            progress=progress,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Predictive results: {output}")


@app.command()
def generative(
    run: str | None = typer.Option(
        None,
        help="Local generative run name or directory; defaults to Yorch233/RSB from Hugging Face Hub.",
    ),
    dataset: str | None = typer.Option(None, help="Registered dataset ID; defaults to the selected ID."),
    split: DatasetSplit = typer.Option(DatasetSplit.TEST, case_sensitive=False),
    sampler: Sampler = typer.Option(Sampler.SDE, case_sensitive=True),
    num_steps: int = typer.Option(5, "--num-steps", min=1),
    skip_type: SkipType = typer.Option(SkipType.UNIFORM, "--skip-type", case_sensitive=True),
    seed: int = typer.Option(10),
    device: str = typer.Option("auto", help="Torch device, CUDA index, or auto."),
    num_workers: int = typer.Option(0, "--num-workers", min=0),
    overwrite: bool = typer.Option(False, help="Replace conflicting or existing result WAV files."),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """Enhance a registered dataset split with a generative RSB model."""
    try:
        output = run_generative_inference(
            run,
            dataset_id=dataset,
            split=split.value,
            sampler=sampler.value,
            num_steps=num_steps,
            skip_type=skip_type.value,
            seed=seed,
            device=device,
            num_workers=num_workers,
            overwrite=overwrite,
            progress=progress,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Generative results: {output}")


if __name__ == "__main__":
    app()
