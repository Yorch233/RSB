"""Shared configuration and launch helpers for training commands."""

from enum import StrEnum
from pathlib import Path
from typing import Any

import typer

from RSB.utils.config import Config
from RSB.utils.paths import DEFAULT_CONFIG_PATH, RUNS_DIR, USER_CONFIG_PATH

_LOCAL_RUNTIME_FIELDS = (
    "dataset",
    "datasets",
    "mixed_precision",
    "multi_gpu",
    "gpu_ids",
    "logger",
    "log_steps",
    "save_state_steps",
    "checkpoints_total_limit",
    "run_dir",
    "results_dir",
    "ncsnpp_operator_backend",
    "ncsnpp_cuda_jit_status",
)


class Optimizer(StrEnum):
    """Supported torch optimizers."""

    ADAM = "Adam"
    ADAMW = "AdamW"


class Logger(StrEnum):
    """Supported experiment loggers."""

    NONE = "none"
    WANDB = "wandb"


def _load_config(
    config_path: Path,
    *,
    use_local_runtime_defaults: bool = False,
    **overrides: Any,
) -> tuple[Config, dict[str, Any]]:
    """Load a run config, merge local runtime settings, and apply CLI overrides."""
    from RSB.utils.config import read_config_from_yaml

    explicit = {
        key: (value.value if isinstance(value, StrEnum) else str(value) if isinstance(value, Path) else value)
        for key, value in overrides.items()
        if value is not None and value != ""
    }
    if overrides.get("run_name") == "":
        explicit["run_name"] = None
    config_path = config_path.expanduser().resolve()
    config = read_config_from_yaml(config_path)
    if use_local_runtime_defaults and USER_CONFIG_PATH.is_file():
        local = read_config_from_yaml(USER_CONFIG_PATH)
        local_values = {field: local.get(field) for field in _LOCAL_RUNTIME_FIELDS if local.get(field) is not None}
        if config_path == DEFAULT_CONFIG_PATH.resolve():
            config.update(local_values)
        else:
            config.update({field: value for field, value in local_values.items() if config.get(field) is None})
    config.update({"run_dir": config.get("run_dir", str(RUNS_DIR))})
    config.update(explicit)
    datasets = config.get("datasets", {})
    if not isinstance(datasets, dict) or not datasets:
        raise typer.BadParameter("No datasets are registered; run 'rsb dataset add --id ID --path PATH'")
    selected_dataset = config.get("dataset")
    if not isinstance(selected_dataset, str) or selected_dataset not in datasets:
        available = ", ".join(datasets)
        raise typer.BadParameter(f"Dataset ID {selected_dataset!r} is not registered; available IDs: {available}")
    return config, explicit


def _validate_resume(resume: bool, checkpoint_path: Path | None) -> None:
    """Validate the resume flag and checkpoint path combination."""
    if resume and checkpoint_path is None:
        raise typer.BadParameter("--checkpoint-path is required with --resume")
    if not resume and checkpoint_path is not None:
        raise typer.BadParameter("--checkpoint-path requires --resume")


def _prepare_training_command(
    config: Config,
    overrides: dict[str, Any],
    *,
    checkpoint_path: Path | None,
    assume_yes: bool,
) -> tuple[Config, dict[str, Any]]:
    """Run rank-zero operator selection and training confirmation."""
    from RSB.training.predictive import _run_path_from_checkpoint
    from RSB.training.preflight import (
        BACKEND_FIELD,
        JIT_STATUS_FIELD,
        TrainingCancelled,
        prepare_training_launch,
    )
    from RSB.training.runtime import distributed_rank
    from RSB.utils.config import read_config_from_yaml

    if distributed_rank() > 0:
        return config, overrides
    if checkpoint_path is not None:
        run_path = _run_path_from_checkpoint(checkpoint_path.resolve())
        config = read_config_from_yaml(run_path / "config.yml")
        config.update(overrides)
    try:
        prepare_training_launch(config, assume_yes=assume_yes)
    except TrainingCancelled as error:
        raise typer.Abort() from error
    overrides.update(
        {
            BACKEND_FIELD: config.get(BACKEND_FIELD),
            JIT_STATUS_FIELD: config.get(JIT_STATUS_FIELD),
        }
    )
    return config, overrides
