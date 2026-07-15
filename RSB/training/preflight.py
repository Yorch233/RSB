"""Interactive preflight checks shared by training commands."""

from __future__ import annotations

from RSB.backbone.ncsnpp.ncsnpp_utils.op.backend import (
    CUDA_JIT_BACKEND,
    PYTORCH_NATIVE_BACKEND,
    enable_cuda_jit,
    use_pytorch_native,
)
from RSB.utils.config import Config

BACKEND_FIELD = "ncsnpp_operator_backend"
JIT_STATUS_FIELD = "ncsnpp_cuda_jit_status"


class TrainingCancelled(RuntimeError):
    """Raised when the user declines a training preflight confirmation."""


def prepare_training_launch(config: Config, *, assume_yes: bool = False) -> None:
    """Select NCSN++ operators, show final parameters, and confirm training."""
    from RSB.cli.tui import get_console, print_config_summary, prompt_yes_no

    requested_backend = config.get(BACKEND_FIELD, CUDA_JIT_BACKEND)
    if requested_backend == PYTORCH_NATIVE_BACKEND:
        use_pytorch_native()
        config.update(
            {
                BACKEND_FIELD: PYTORCH_NATIVE_BACKEND,
                JIT_STATUS_FIELD: config.get(JIT_STATUS_FIELD, "not_attempted"),
            }
        )
    elif requested_backend == CUDA_JIT_BACKEND:
        try:
            enable_cuda_jit()
        except Exception as error:
            config.update({JIT_STATUS_FIELD: "failed"})
            get_console().print(f"[error]CUDA JIT initialization failed:[/error] {error}")
            if not assume_yes and not prompt_yes_no(
                "Continue training with native PyTorch NCSN++ operators?",
                default=True,
            ):
                raise TrainingCancelled("Native PyTorch operator fallback was declined") from error
            use_pytorch_native()
            config.update({BACKEND_FIELD: PYTORCH_NATIVE_BACKEND})
        else:
            config.update({BACKEND_FIELD: CUDA_JIT_BACKEND, JIT_STATUS_FIELD: "available"})
    else:
        raise ValueError(f"Unsupported NCSN++ operator backend: {requested_backend!r}")

    print_config_summary(config.dict(), title="Training parameters", boxed=False)
    if not assume_yes and not prompt_yes_no("Start training with these parameters?", default=True):
        raise TrainingCancelled("Training was declined")
