"""Interactive configuration wizard for RSB training."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import typer

from RSB.cli.tui import (
    get_console,
    print_config_summary,
    prompt_text,
    prompt_validated_text,
    prompt_yes_no,
    rich_select,
)
from RSB.data.dataset_registry import (
    add_dataset_entry,
    edit_dataset_entry,
    normalize_dataset_registry,
    remove_dataset_entry,
    validate_dataset_id,
    validate_dataset_root,
)
from RSB.utils.config import CURRENT_CONFIG_VERSION, BaseConfiguer, read_yml
from RSB.utils.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT, RESULTS_DIR, RUNS_DIR, USER_CONFIG_PATH

PRECISIONS = ("none", "fp16", "bf16")
ADD_DATASET = "＋ Add dataset"
SAVE_DATASETS = "✓ Save and continue"


def _resolve_project_path(path: Path) -> Path:
    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else PROJECT_ROOT / expanded).resolve()


def parse_gpu_ids(value: str, *, gpu_count: int, multi_gpu: bool) -> str | list[int]:
    """Parse and validate a comma/space-separated GPU selection."""
    if gpu_count < 1:
        raise ValueError("No CUDA GPUs were detected; RSB training requires at least one GPU")
    normalized = value.strip().lower()
    if normalized == "all":
        if multi_gpu and gpu_count == 1:
            raise ValueError("Multi-GPU training requires at least two GPUs")
        return "all"

    tokens = [token for token in re.split(r"[\s,]+", normalized) if token]
    if not tokens or any(not token.isdigit() for token in tokens):
        raise ValueError("GPU IDs must be 'all' or comma-separated non-negative integers")
    device_ids = [int(token) for token in tokens]
    if len(device_ids) != len(set(device_ids)):
        raise ValueError("GPU IDs must not contain duplicates")
    if gpu_count > 0 and any(device_id >= gpu_count for device_id in device_ids):
        raise ValueError(f"GPU IDs must be smaller than the detected GPU count ({gpu_count})")
    if multi_gpu and len(device_ids) < 2:
        raise ValueError("Multi-GPU training requires at least two GPU IDs")
    if not multi_gpu and len(device_ids) != 1:
        raise ValueError("Single-GPU training requires exactly one GPU ID")
    return device_ids


def _select(prompt: str, choices: list[str], default: int) -> str:
    return choices[rich_select(prompt, choices, default=default)]


def _prompt_new_dataset(datasets: dict[str, str]) -> dict[str, str]:
    def validate_new_id(value: str) -> str:
        """Validate a new, unique registry ID."""
        normalized = validate_dataset_id(value)
        if normalized in datasets:
            raise ValueError(f"Dataset ID {normalized!r} is already registered")
        return normalized

    dataset_id = prompt_validated_text(
        "What ID should identify this dataset",
        validate_new_id,
        description="Use this ID with the training --dataset option.",
        input_hint="Enter a unique ID using letters, numbers, '.', '_', or '-'.",
    )
    dataset_path = prompt_validated_text(
        "Where is the paired dataset directory",
        lambda value: validate_dataset_root(Path(value)),
        description="Must contain {train, valid, test}/{clean, noisy}.",
        input_hint="Enter the dataset root path.",
    )
    return add_dataset_entry(datasets, dataset_id, dataset_path)


def _prompt_dataset_edit(datasets: dict[str, str], dataset_id: str) -> dict[str, str]:
    def validate_replacement_id(value: str) -> str:
        """Validate an edited registry ID without colliding with another entry."""
        normalized = validate_dataset_id(value)
        if normalized != dataset_id and normalized in datasets:
            raise ValueError(f"Dataset ID {normalized!r} is already registered")
        return normalized

    replacement_id = prompt_validated_text(
        "What should the dataset ID be",
        validate_replacement_id,
        default=dataset_id,
        description="This ID is passed to training commands.",
    )
    replacement_path = prompt_validated_text(
        "Where is the paired dataset directory",
        lambda value: validate_dataset_root(Path(value)),
        default=datasets[dataset_id],
        description="Must contain {train, valid, test}/{clean, noisy}.",
        input_hint="Enter a replacement path, or press Enter to keep the current path.",
    )
    return edit_dataset_entry(datasets, dataset_id, new_id=replacement_id, path=replacement_path)


def _manage_datasets(initial: Mapping[str, Any]) -> dict[str, str]:
    datasets = normalize_dataset_registry(dict(initial))
    while True:
        dataset_ids = list(datasets)
        entries = [f"{dataset_id}  {datasets[dataset_id]}" for dataset_id in dataset_ids]
        choices = [*entries, ADD_DATASET, SAVE_DATASETS]
        default = len(choices) - 1 if datasets else 0
        selected = rich_select("Which dataset registry item do you want to manage", choices, default=default)
        if selected == len(dataset_ids):
            datasets = _prompt_new_dataset(datasets)
            continue
        if selected == len(dataset_ids) + 1:
            return datasets

        dataset_id = dataset_ids[selected]
        action = _select("What do you want to do with this dataset", ["Edit", "Remove", "Cancel"], 2)
        if action == "Edit":
            datasets = _prompt_dataset_edit(datasets, dataset_id)
        elif action == "Remove" and prompt_yes_no(f"Remove dataset registration {dataset_id!r}", default=False):
            datasets = remove_dataset_entry(datasets, dataset_id)


def _configure_datasets(current: Mapping[str, Any]) -> tuple[dict[str, str], str | None]:
    datasets = normalize_dataset_registry(current.get("datasets"))
    if prompt_yes_no("Configure the dataset registry", default=not datasets):
        datasets = _manage_datasets(datasets)
    if not datasets:
        return datasets, None
    dataset_ids = list(datasets)
    configured_id = current.get("dataset")
    default = dataset_ids.index(configured_id) if isinstance(configured_id, str) and configured_id in datasets else 0
    selected_id = _select("Which dataset should training use", dataset_ids, default)
    return datasets, selected_id


def _prompt_positive_int(prompt: str, default: int) -> int:
    def validate(value: str) -> int:
        """Parse a positive integer."""
        try:
            parsed = int(value)
        except ValueError as error:
            raise ValueError("Please enter a valid integer.") from error
        if parsed <= 0:
            raise ValueError("Value must be greater than zero.")
        return parsed

    return prompt_validated_text(
        prompt,
        validate,
        default=str(default),
        input_hint="Enter a positive integer, or press Enter to use the default.",
    )


def _prompt_devices(gpu_count: int, multi_gpu: bool, default: str | list[int]) -> str | list[int]:
    displayed_default = default if isinstance(default, str) else ",".join(str(device_id) for device_id in default)
    return prompt_validated_text(
        "GPU IDs",
        lambda value: parse_gpu_ids(value, gpu_count=gpu_count, multi_gpu=multi_gpu),
        default=displayed_default,
        description="Use comma-separated GPU IDs, or 'all'.",
        input_hint="Enter the GPU selection, or press Enter to use the default.",
    )


def _choice_index(choices: tuple[str, ...] | list[str], value: Any, fallback: int) -> int:
    return choices.index(value) if value in choices else fallback


def _positive_default(defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    value = defaults.get(key, fallback)
    return value if type(value) is int and value > 0 else fallback


def collect_configuration(defaults: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Interactively collect dataset, runtime, logging, and checkpoint settings."""
    current = defaults or {}
    console = get_console()
    console.print("[accent]▱[/accent] [bold]RSB Configuration Wizard[/bold]")
    console.print("[dim]│[/dim]")
    gpu_count = torch.cuda.device_count()
    console.print(f"[info]◆[/info] Detected GPUs: {gpu_count}")
    console.print("[dim]│[/dim]")
    if gpu_count < 1:
        console.print("[error]No CUDA GPUs were detected. Configure RSB on a GPU training host.[/error]")
        raise typer.Abort
    datasets, dataset_id = _configure_datasets(current)
    precision_name = _select(
        "Mixed precision",
        list(PRECISIONS),
        _choice_index(PRECISIONS, current.get("mixed_precision"), 0),
    )
    configured_multi_gpu = current.get("multi_gpu")
    multi_gpu = prompt_yes_no(
        "Enable multi-GPU training",
        default=configured_multi_gpu if isinstance(configured_multi_gpu, bool) else gpu_count > 1,
    )
    configured_gpu_ids = current.get("gpu_ids", "all")
    if not isinstance(configured_gpu_ids, (str, list)):
        configured_gpu_ids = "all"
    devices = _prompt_devices(gpu_count, multi_gpu, configured_gpu_ids)
    loggers = ["wandb", "none"]
    logger = _select("Experiment logger", loggers, _choice_index(loggers, current.get("logger"), 0))
    log_steps = _prompt_positive_int("Log training loss every N steps", _positive_default(current, "log_steps", 10))
    save_state_steps = _prompt_positive_int(
        "Save resumable state every N steps",
        _positive_default(current, "save_state_steps", 1000),
    )
    checkpoints_total_limit = _prompt_positive_int(
        "Intermediate checkpoint limit",
        _positive_default(current, "checkpoints_total_limit", 3),
    )
    default_run_dir = _resolve_project_path(Path(str(current.get("run_dir", RUNS_DIR))))
    run_dir = str(_resolve_project_path(Path(prompt_text("Run directory", str(default_run_dir)))))
    results_dir = str(_resolve_project_path(Path(str(current.get("results_dir", RESULTS_DIR)))))

    return {
        "dataset": dataset_id,
        "datasets": datasets,
        "mixed_precision": precision_name,
        "multi_gpu": multi_gpu,
        "gpu_ids": devices,
        "logger": logger,
        "log_steps": log_steps,
        "save_state_steps": save_state_steps,
        "checkpoints_total_limit": checkpoints_total_limit,
        "run_dir": run_dir,
        "results_dir": results_dir,
    }


def write_configuration(configuration: dict[str, Any], output_path: Path) -> Path:
    """Write a wizard configuration that inherits model defaults."""
    destination = _resolve_project_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    inherited = os.path.relpath(DEFAULT_CONFIG_PATH.resolve(), destination.parent)
    BaseConfiguer.dump({"inherit": inherited, "version": CURRENT_CONFIG_VERSION, **configuration}, destination)
    return destination


def configure(
    output_path: Path = typer.Option(USER_CONFIG_PATH, "--output", dir_okay=False),
    force: bool = typer.Option(False, "--force", help="Overwrite and save without confirmation."),
) -> None:
    """Launch the interactive RSB configuration wizard."""
    destination = _resolve_project_path(output_path)
    if destination.exists() and not force and not prompt_yes_no(f"Overwrite {destination}?", default=False):
        raise typer.Abort
    defaults = read_yml(destination) if destination.exists() else {}
    configuration = collect_configuration(defaults)
    print_config_summary(
        {**configuration, "config_path": str(destination)},
        title="RSB configuration",
        boxed=False,
    )
    if not force and not prompt_yes_no("Save this configuration?", default=True):
        raise typer.Abort
    write_configuration(configuration, destination)
    get_console().print("[success]◆[/success] Configuration saved.")
