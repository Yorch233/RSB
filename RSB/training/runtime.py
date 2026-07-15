"""Resolve user-facing training settings into Lightning Trainer arguments."""

import os
from dataclasses import dataclass

from RSB.utils.config import Config

MIXED_PRECISION = {"none": "32-true", "fp16": "16-mixed", "bf16": "bf16-mixed"}


def distributed_rank() -> int:
    """Return the rank assigned by a distributed launcher, defaulting to zero."""
    value = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Distributed rank must be an integer, got {value!r}") from error


@dataclass(frozen=True)
class TrainerRuntime:
    """Lightning runtime values resolved from local wizard settings."""

    accelerator: str
    devices: str | int | list[int]
    strategy: str
    precision: str


def resolve_trainer_runtime(config: Config) -> TrainerRuntime:
    """Resolve wizard fields while honoring explicit advanced CLI overrides."""
    mixed_precision = config.get("mixed_precision", "none")
    if mixed_precision not in MIXED_PRECISION:
        choices = ", ".join(MIXED_PRECISION)
        raise ValueError(f"mixed_precision must be one of: {choices}")
    precision = config.get("precision") or MIXED_PRECISION[mixed_precision]

    multi_gpu = config.get("multi_gpu", False)
    if not isinstance(multi_gpu, bool):
        raise ValueError("multi_gpu must be a boolean")
    gpu_ids = config.get("gpu_ids", "all")
    if gpu_ids == "all":
        devices: str | int | list[int] = "auto" if multi_gpu else 1
    elif isinstance(gpu_ids, list) and gpu_ids and all(isinstance(device_id, int) for device_id in gpu_ids):
        if len(gpu_ids) != len(set(gpu_ids)) or any(device_id < 0 for device_id in gpu_ids):
            raise ValueError("gpu_ids must contain unique, non-negative integer IDs")
        if multi_gpu and len(gpu_ids) < 2:
            raise ValueError("multi_gpu requires at least two GPU IDs")
        if not multi_gpu and len(gpu_ids) != 1:
            raise ValueError("single-GPU training requires exactly one GPU ID")
        devices = gpu_ids
    else:
        raise ValueError("gpu_ids must be 'all' or a non-empty list of integer IDs")

    return TrainerRuntime(
        accelerator=config.get("accelerator", "gpu"),
        devices=config.get("devices") or devices,
        strategy=config.get("strategy") or ("ddp" if multi_gpu else "auto"),
        precision=precision,
    )
