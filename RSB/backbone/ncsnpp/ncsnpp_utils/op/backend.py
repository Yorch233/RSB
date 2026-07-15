"""Select and initialize the NCSN++ operator implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

CUDA_JIT_BACKEND = "cuda_jit"
PYTORCH_NATIVE_BACKEND = "pytorch_native"

_backend = PYTORCH_NATIVE_BACKEND
_fused_extension: Any | None = None
_upfirdn_extension: Any | None = None


def operator_backend() -> str:
    """Return the active NCSN++ operator backend."""
    return _backend


def use_pytorch_native() -> None:
    """Use portable PyTorch operators with CPU/CUDA autograd support."""
    global _backend
    _backend = PYTORCH_NATIVE_BACKEND


def enable_cuda_jit() -> None:
    """Compile and activate the optimized NCSN++ CUDA extensions."""
    global _backend, _fused_extension, _upfirdn_extension
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    module_path = Path(__file__).resolve().parent
    if _fused_extension is None:
        _fused_extension = load(
            "rsb_ncsnpp_fused_bias_act",
            sources=[
                str(module_path / "fused_bias_act.cpp"),
                str(module_path / "fused_bias_act_kernel.cu"),
            ],
        )
    if _upfirdn_extension is None:
        _upfirdn_extension = load(
            "rsb_ncsnpp_upfirdn2d",
            sources=[
                str(module_path / "upfirdn2d.cpp"),
                str(module_path / "upfirdn2d_kernel.cu"),
            ],
        )
    _backend = CUDA_JIT_BACKEND


def activate_operator_backend(backend: str) -> None:
    """Activate a persisted NCSN++ operator backend."""
    if backend == CUDA_JIT_BACKEND:
        enable_cuda_jit()
        return
    if backend == PYTORCH_NATIVE_BACKEND:
        use_pytorch_native()
        return
    raise ValueError(f"Unsupported NCSN++ operator backend: {backend!r}")


def fused_extension() -> Any:
    """Return the loaded fused-bias activation extension."""
    if _backend != CUDA_JIT_BACKEND or _fused_extension is None:
        raise RuntimeError("The NCSN++ fused CUDA JIT backend is not active")
    return _fused_extension


def upfirdn_extension() -> Any:
    """Return the loaded upfirdn extension."""
    if _backend != CUDA_JIT_BACKEND or _upfirdn_extension is None:
        raise RuntimeError("The NCSN++ upfirdn CUDA JIT backend is not active")
    return _upfirdn_extension
