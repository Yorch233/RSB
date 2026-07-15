"""Fused-bias activation with CUDA JIT and native PyTorch backends."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch.nn import functional as F

from RSB.backbone.ncsnpp.ncsnpp_utils.op.backend import (
    CUDA_JIT_BACKEND,
    fused_extension,
    operator_backend,
)


class _FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(
        ctx: Any,
        grad_output: Tensor,
        output: Tensor,
        negative_slope: float,
        scale: float,
    ) -> tuple[Tensor, Tensor]:
        ctx.save_for_backward(output)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        empty = grad_output.new_empty(0)
        grad_input = fused_extension().fused_bias_act(
            grad_output,
            empty,
            output,
            3,
            1,
            negative_slope,
            scale,
        )
        dimensions = [0]
        if grad_input.ndim > 2:
            dimensions += list(range(2, grad_input.ndim))
        return grad_input, grad_input.sum(dimensions).detach()

    @staticmethod
    def backward(ctx: Any, gradgrad_input: Tensor, gradgrad_bias: Tensor) -> tuple[Tensor, None, None, None]:
        (output,) = ctx.saved_tensors
        gradgrad_output = fused_extension().fused_bias_act(
            gradgrad_input,
            gradgrad_bias,
            output,
            3,
            1,
            ctx.negative_slope,
            ctx.scale,
        )
        return gradgrad_output, None, None, None


class _FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, bias: Tensor, negative_slope: float, scale: float) -> Tensor:
        empty = input.new_empty(0)
        output = fused_extension().fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(output)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, Tensor, None, None]:
        (output,) = ctx.saved_tensors
        grad_input, grad_bias = _FusedLeakyReLUFunctionBackward.apply(
            grad_output,
            output,
            ctx.negative_slope,
            ctx.scale,
        )
        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    """Apply a learned channel bias followed by scaled leaky ReLU."""

    def __init__(self, channel: int, negative_slope: float = 0.2, scale: float = 2**0.5) -> None:
        """Initialize the channel bias and activation parameters."""
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input: Tensor) -> Tensor:
        """Apply the biased activation using the active operator backend."""
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(
    input: Tensor,
    bias: Tensor,
    negative_slope: float = 0.2,
    scale: float = 2**0.5,
) -> Tensor:
    """Apply fused CUDA or portable native biased activation."""
    if input.device.type == "cuda" and operator_backend() == CUDA_JIT_BACKEND:
        return _FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
    rest_dimensions = [1] * (input.ndim - bias.ndim - 1)
    shifted = input + bias.view(1, bias.shape[0], *rest_dimensions)
    return F.leaky_relu(shifted, negative_slope=negative_slope) * scale
