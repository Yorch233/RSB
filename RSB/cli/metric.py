"""Metric calculation command for manifested inference directories."""

import sys
from pathlib import Path

import typer

from RSB.cli.tui import rich_select
from RSB.workflows.artifacts import resolve_results_root, resolve_run
from RSB.workflows.evaluation import DEFAULT_METRICS, SUPPORTED_METRICS, evaluate_directory, evaluate_external


def _parse_metrics(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_METRICS
    metrics = tuple(
        metric.strip().lower().replace("-", "_") for value in values for metric in value.split(",") if metric.strip()
    )
    invalid = [metric for metric in metrics if metric not in SUPPORTED_METRICS]
    if invalid:
        raise typer.BadParameter(f"Unsupported metrics: {invalid}; choose from {SUPPORTED_METRICS}")
    return tuple(dict.fromkeys(metrics))


def _run_result_directory(run_reference: str, result: str | None) -> Path:
    run = resolve_run(run_reference)
    root = resolve_results_root(run.config) / run.name
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and (path / "inference.json").is_file())
    if not candidates:
        raise ValueError(f"Run has no manifested inference results: {root}")
    if result is not None:
        selected = root / result
        if selected not in candidates:
            available = ", ".join(path.name for path in candidates)
            raise ValueError(f"Result {result!r} is not available; choose from: {available}")
        return selected
    if len(candidates) == 1:
        return candidates[0]
    if not sys.stdin.isatty():
        available = ", ".join(path.name for path in candidates)
        raise ValueError(f"Multiple results are available; pass --result with one of: {available}")
    selected = rich_select("Which inference result should be evaluated", [path.name for path in candidates])
    return candidates[selected]


def calculate(
    directory: Path | None = typer.Option(
        None,
        "--dir",
        exists=True,
        file_okay=False,
        help="Manifested inference result directory.",
    ),
    run: str | None = typer.Option(None, help="Run name or path whose result should be selected."),
    result: str | None = typer.Option(None, help="Result directory name, such as predictive or SDE_N=50."),
    clean: Path | None = typer.Option(None, exists=True, file_okay=False, help="Third-party clean WAV directory."),
    noisy: Path | None = typer.Option(None, exists=True, file_okay=False, help="Third-party noisy WAV directory."),
    enhanced: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
        help="Third-party enhanced WAV directory and metric output location.",
    ),
    metrics: list[str] | None = typer.Option(
        None,
        "--metrics",
        "--metric",
        help="Metric name or comma-separated names; repeatable. Defaults to PESQ, ESTOI, and SI-SDR.",
    ),
    sample_rate: int = typer.Option(16_000, "--sample-rate", min=1, help="Third-party audio sample rate."),
    max_workers: int = typer.Option(0, "--max-workers", min=0),
    overwrite: bool = typer.Option(False, help="Recalculate existing metric artifacts."),
) -> None:
    """Calculate metrics for a result directory, a run result, or third-party WAV directories."""
    selected = _parse_metrics(metrics)
    try:
        external_paths = (clean, noisy, enhanced)
        external_mode = any(path is not None for path in external_paths)
        modes = int(directory is not None) + int(run is not None) + int(external_mode)
        if modes != 1:
            raise ValueError("Choose exactly one input mode: --dir, --run, or --clean/--noisy/--enhanced")
        if result is not None and run is None:
            raise ValueError("--result requires --run")
        if external_mode and not all(path is not None for path in external_paths):
            raise ValueError("Third-party evaluation requires --clean, --noisy, and --enhanced together")
        if run is not None:
            directory = _run_result_directory(run, result)
        if directory is not None:
            csv_path, json_path = evaluate_directory(
                directory,
                metrics=selected,
                max_workers=max_workers,
                overwrite=overwrite,
            )
        else:
            assert clean is not None
            assert noisy is not None
            assert enhanced is not None
            csv_path, json_path = evaluate_external(
                clean,
                noisy,
                enhanced,
                metrics=selected,
                sample_rate=sample_rate,
                max_workers=max_workers,
                overwrite=overwrite,
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Per-file metrics: {csv_path}")
    typer.echo(f"Metric summary: {json_path}")
