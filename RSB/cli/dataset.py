from pathlib import Path

import typer
from rich.table import Table

from RSB.cli.tui import get_console
from RSB.cli.tui.theme import ACCENT, TEXT
from RSB.data.create_dataset import CleanDataset, NoiseDataset, create_dataset
from RSB.data.dataset_registry import (
    add_dataset_entry,
    edit_dataset_entry,
    load_dataset_registry,
    remove_dataset_entry,
    save_dataset_registry,
)
from RSB.utils.paths import USER_CONFIG_PATH
from RSB.workflows.artifacts import SPLITS

app = typer.Typer(help="Create and inspect paired speech datasets.")


def _registry_error(error: KeyError | TypeError | ValueError) -> typer.BadParameter:
    message = error.args[0] if isinstance(error, KeyError) else str(error)
    return typer.BadParameter(str(message))


@app.command("list")
def list_registered() -> None:
    """List datasets registered in the project configuration."""
    try:
        state = load_dataset_registry(USER_CONFIG_PATH)
    except (TypeError, ValueError) as error:
        raise _registry_error(error) from error
    if not state.datasets:
        typer.echo("No datasets registered.")
        return
    table = Table(show_header=True, header_style=f"bold {ACCENT}", box=None)
    table.add_column("ID", style=f"bold {TEXT}")
    table.add_column("Path", style=TEXT, overflow="fold")
    table.add_column("Training", justify="center")
    for dataset_id, path in state.datasets.items():
        table.add_row(dataset_id, path, "●" if dataset_id == state.selected_id else "")
    get_console().print(table)


@app.command("add")
def add_registered(
    dataset_id: str = typer.Option(..., "--id", help="Unique dataset ID used by training commands."),
    path: Path = typer.Option(..., "--path", file_okay=False, help="Paired dataset root directory."),
    select: bool = typer.Option(False, "--select", help="Select this ID as the default training dataset."),
) -> None:
    """Register a paired dataset by ID and path."""
    try:
        state = load_dataset_registry(USER_CONFIG_PATH)
        datasets = add_dataset_entry(state.datasets, dataset_id, path)
        selected_id = dataset_id.strip() if select or state.selected_id is None else state.selected_id
        save_dataset_registry(datasets, selected_id, USER_CONFIG_PATH)
    except (KeyError, TypeError, ValueError) as error:
        raise _registry_error(error) from error
    typer.echo(f"Registered dataset {dataset_id.strip()!r}.")


@app.command("edit")
def edit_registered(
    dataset_id: str = typer.Option(..., "--id", help="Existing registered dataset ID."),
    new_id: str | None = typer.Option(None, "--new-id", help="Replacement dataset ID."),
    path: Path | None = typer.Option(None, "--path", file_okay=False, help="Replacement dataset root."),
    select: bool = typer.Option(False, "--select", help="Select the edited ID for training."),
) -> None:
    """Edit a registered dataset ID and/or path."""
    if new_id is None and path is None and not select:
        raise typer.BadParameter("Provide --new-id, --path, and/or --select")
    try:
        state = load_dataset_registry(USER_CONFIG_PATH)
        datasets = edit_dataset_entry(state.datasets, dataset_id, new_id=new_id, path=path)
        replacement_id = (new_id or dataset_id).strip()
        selected_id = state.selected_id
        if selected_id == dataset_id or select:
            selected_id = replacement_id
        save_dataset_registry(datasets, selected_id, USER_CONFIG_PATH)
    except (KeyError, TypeError, ValueError) as error:
        raise _registry_error(error) from error
    typer.echo(f"Updated dataset {dataset_id!r} as {replacement_id!r}.")


@app.command("delete")
def delete_registered(
    dataset_id: str = typer.Option(..., "--id", help="Registered dataset ID to remove."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Delete without an interactive confirmation."),
) -> None:
    """Remove a dataset registration without deleting its files."""
    try:
        state = load_dataset_registry(USER_CONFIG_PATH)
        if dataset_id not in state.datasets:
            raise KeyError(f"Dataset ID {dataset_id!r} is not registered")
        if not yes and not typer.confirm(f"Remove dataset registration {dataset_id!r}?"):
            raise typer.Abort
        datasets = remove_dataset_entry(state.datasets, dataset_id)
        selected_id = state.selected_id
        if selected_id == dataset_id:
            selected_id = next(iter(datasets), None)
        save_dataset_registry(datasets, selected_id, USER_CONFIG_PATH)
    except (KeyError, TypeError, ValueError) as error:
        raise _registry_error(error) from error
    typer.echo(f"Removed dataset registration {dataset_id!r}; files were not deleted.")


@app.command()
def create(
    task: list[str] = typer.Option(..., "--task", help="Repeat for enhancement and/or dereverberation."),
    clean: tuple[CleanDataset, Path] = typer.Option(..., "--clean", help="Clean dataset TYPE and PATH."),
    noise: tuple[NoiseDataset, Path] = typer.Option(..., "--noise", help="Noise dataset TYPE and PATH."),
    output_dir: Path = typer.Option(..., "--output_dir", "--output-dir", help="Dataset output directory."),
    sample_rate: int = typer.Option(16_000, min=1),
    snr_min: float = typer.Option(-6.0),
    snr_max: float = typer.Option(14.0),
    t60_min: float = typer.Option(0.4, min=0.01),
    t60_max: float = typer.Option(1.0, min=0.01),
    seed: int = typer.Option(100),
    overwrite: bool = typer.Option(False, help="Replace a non-empty output directory."),
) -> None:
    """Create paired clean/noisy splits and export synthesis statistics."""
    try:
        configuration = create_dataset(
            tasks=task,
            clean_dataset=clean[0],
            clean_inputs=[clean[1]],
            noise_dataset=noise[0],
            noise_inputs=[noise[1]],
            output_dir=output_dir,
            sample_rate=sample_rate,
            snr_range_db=(snr_min, snr_max),
            t60_range_s=(t60_min, t60_max),
            seed=seed,
            overwrite=overwrite,
        )
    except (FileExistsError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error

    summary = configuration["summary"]
    typer.echo(f"Created {summary['num_files']} pairs in {output_dir.expanduser().resolve()}")
    typer.echo(f"Configuration: {output_dir.expanduser().resolve() / 'create_configuraton.json'}")


@app.command()
def inspect(directory: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    """Report paired WAV counts in an exported dataset."""
    for split in ("train", "valid", "test"):
        clean_count = len(list((directory / split / "clean").glob("*.wav")))
        noisy_count = len(list((directory / split / "noisy").glob("*.wav")))
        typer.echo(f"{split}: clean={clean_count}, noisy={noisy_count}")


@app.command("generate-mean")
def generate_mean(
    run: str = typer.Option(..., help="Predictive run name or directory."),
    dataset: str | None = typer.Option(None, help="Registered dataset ID; defaults to the selected ID."),
    split: list[str] | None = typer.Option(None, "--split", help="Repeat to override the default of all splits."),
    device: str = typer.Option("auto", help="Torch device, CUDA index, or auto."),
    num_workers: int = typer.Option(0, "--num-workers", min=0),
    overwrite: bool = typer.Option(False, help="Replace existing posterior means."),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """Generate offline predictive posterior means inside a registered dataset."""
    from RSB.workflows.posterior import generate_posterior_means

    try:
        manifests = generate_posterior_means(
            run,
            dataset_id=dataset,
            splits=tuple(split or SPLITS),
            device=device,
            num_workers=num_workers,
            overwrite=overwrite,
            progress=progress,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    for manifest in manifests:
        typer.echo(f"Posterior means: {manifest.parent}")
