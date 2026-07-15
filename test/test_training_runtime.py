import pytest

from RSB.training.runtime import resolve_trainer_runtime
from RSB.utils.config import Config


@pytest.mark.parametrize(
    ("mixed_precision", "expected"),
    [("none", "32-true"), ("fp16", "16-mixed"), ("bf16", "bf16-mixed")],
)
def test_runtime_maps_wizard_precision(mixed_precision: str, expected: str) -> None:
    runtime = resolve_trainer_runtime(Config({"mixed_precision": mixed_precision}))
    assert runtime.precision == expected


def test_runtime_defaults_to_full_precision() -> None:
    runtime = resolve_trainer_runtime(Config({}))
    assert runtime.precision == "32-true"


def test_runtime_maps_multi_gpu_all_to_ddp_auto_devices() -> None:
    runtime = resolve_trainer_runtime(Config({"multi_gpu": True, "gpu_ids": "all"}))
    assert runtime.accelerator == "gpu"
    assert runtime.devices == "auto"
    assert runtime.strategy == "ddp"


def test_runtime_preserves_explicit_gpu_ids_and_advanced_overrides() -> None:
    runtime = resolve_trainer_runtime(
        Config(
            {
                "mixed_precision": "none",
                "multi_gpu": True,
                "gpu_ids": [1, 3],
                "precision": "64-true",
                "accelerator": "cpu",
                "devices": 2,
                "strategy": "auto",
            }
        )
    )
    assert runtime.precision == "64-true"
    assert runtime.accelerator == "cpu"
    assert runtime.devices == 2
    assert runtime.strategy == "auto"


@pytest.mark.parametrize(
    "configuration",
    [
        {"multi_gpu": "yes", "gpu_ids": "all"},
        {"multi_gpu": True, "gpu_ids": [0]},
        {"multi_gpu": False, "gpu_ids": [0, 1]},
        {"multi_gpu": True, "gpu_ids": [0, 0]},
        {"multi_gpu": False, "gpu_ids": [-1]},
    ],
)
def test_runtime_rejects_inconsistent_device_configuration(configuration: dict) -> None:
    with pytest.raises(ValueError, match="multi_gpu|GPU|gpu_ids|single-GPU"):
        resolve_trainer_runtime(Config(configuration))
