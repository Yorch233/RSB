"""Upsample-filter-downsample with CUDA JIT and native PyTorch backends."""

from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.autograd import Function
from torch.nn import functional as F

from RSB.backbone.ncsnpp.ncsnpp_utils.op.backend import (
    CUDA_JIT_BACKEND,
    operator_backend,
    upfirdn_extension,
)


class _UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx: Any,
        grad_output: Tensor,
        kernel: Tensor,
        grad_kernel: Tensor,
        up: tuple[int, int],
        down: tuple[int, int],
        pad: tuple[int, int, int, int],
        grad_pad: tuple[int, int, int, int],
        input_size: tuple[int, ...],
        output_size: tuple[int, int],
    ) -> Tensor:
        up_x, up_y = up
        down_x, down_y = down
        grad_pad_x0, grad_pad_x1, grad_pad_y0, grad_pad_y1 = grad_pad
        reshaped = grad_output.reshape(-1, output_size[0], output_size[1], 1)
        grad_input = upfirdn_extension().upfirdn2d(
            reshaped,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            grad_pad_x0,
            grad_pad_x1,
            grad_pad_y0,
            grad_pad_y1,
        )
        grad_input = grad_input.view(input_size)
        ctx.save_for_backward(kernel)
        ctx.up = up
        ctx.down = down
        ctx.pad = pad
        ctx.input_size = input_size
        ctx.output_size = output_size
        return grad_input

    @staticmethod
    def backward(ctx: Any, gradgrad_input: Tensor) -> tuple[Tensor, None, None, None, None, None, None, None, None]:
        (kernel,) = ctx.saved_tensors
        up_x, up_y = ctx.up
        down_x, down_y = ctx.down
        pad_x0, pad_x1, pad_y0, pad_y1 = ctx.pad
        reshaped = gradgrad_input.reshape(-1, ctx.input_size[2], ctx.input_size[3], 1)
        gradgrad_output = upfirdn_extension().upfirdn2d(
            reshaped,
            kernel,
            up_x,
            up_y,
            down_x,
            down_y,
            pad_x0,
            pad_x1,
            pad_y0,
            pad_y1,
        )
        gradgrad_output = gradgrad_output.view(
            ctx.input_size[0],
            ctx.input_size[1],
            ctx.output_size[0],
            ctx.output_size[1],
        )
        return gradgrad_output, None, None, None, None, None, None, None, None


class _UpFirDn2d(Function):
    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        kernel: Tensor,
        up: tuple[int, int],
        down: tuple[int, int],
        pad: tuple[int, int, int, int],
    ) -> Tensor:
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        kernel_height, kernel_width = kernel.shape
        _, channel, input_height, input_width = input.shape
        ctx.input_size = tuple(input.shape)
        reshaped = input.reshape(-1, input_height, input_width, 1)
        ctx.save_for_backward(kernel, kernel.flip([0, 1]))
        output_height = (input_height * up_y + pad_y0 + pad_y1 - kernel_height) // down_y + 1
        output_width = (input_width * up_x + pad_x0 + pad_x1 - kernel_width) // down_x + 1
        ctx.output_size = (output_height, output_width)
        ctx.up = up
        ctx.down = down
        ctx.pad = pad
        ctx.grad_pad = (
            kernel_width - pad_x0 - 1,
            input_width * up_x - output_width * down_x + pad_x0 - up_x + 1,
            kernel_height - pad_y0 - 1,
            input_height * up_y - output_height * down_y + pad_y0 - up_y + 1,
        )
        output = upfirdn_extension().upfirdn2d(
            reshaped,
            kernel,
            up_x,
            up_y,
            down_x,
            down_y,
            pad_x0,
            pad_x1,
            pad_y0,
            pad_y1,
        )
        return output.view(-1, channel, output_height, output_width)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None, None, None]:
        kernel, grad_kernel = ctx.saved_tensors
        grad_input = _UpFirDn2dBackward.apply(
            grad_output,
            kernel,
            grad_kernel,
            ctx.up,
            ctx.down,
            ctx.pad,
            ctx.grad_pad,
            ctx.input_size,
            ctx.output_size,
        )
        return grad_input, None, None, None, None


def upfirdn2d(
    input: Tensor,
    kernel: Tensor,
    up: int = 1,
    down: int = 1,
    pad: tuple[int, int] = (0, 0),
) -> Tensor:
    """Upsample, pad, filter, and downsample a batched feature map."""
    if input.device.type == "cuda" and operator_backend() == CUDA_JIT_BACKEND:
        return _UpFirDn2d.apply(input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1]))
    return upfirdn2d_native(
        input,
        kernel,
        up,
        up,
        down,
        down,
        pad[0],
        pad[1],
        pad[0],
        pad[1],
    )


def upfirdn2d_native(
    input: Tensor,
    kernel: Tensor,
    up_x: int,
    up_y: int,
    down_x: int,
    down_y: int,
    pad_x0: int,
    pad_x1: int,
    pad_y0: int,
    pad_y1: int,
) -> Tensor:
    """Run upfirdn with differentiable operators supported on CPU and GPU."""
    _, channel, input_height, input_width = input.shape
    input = input.reshape(-1, input_height, input_width, 1)
    _, input_height, input_width, minor = input.shape
    kernel_height, kernel_width = kernel.shape
    output = input.view(-1, input_height, 1, input_width, 1, minor)
    output = F.pad(output, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    output = output.view(-1, input_height * up_y, input_width * up_x, minor)
    output = F.pad(output, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    output = output[
        :,
        max(-pad_y0, 0) : output.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : output.shape[2] - max(-pad_x1, 0),
        :,
    ]
    output = output.permute(0, 3, 1, 2)
    output = output.reshape(
        -1,
        1,
        input_height * up_y + pad_y0 + pad_y1,
        input_width * up_x + pad_x0 + pad_x1,
    )
    weight = kernel.flip([0, 1]).view(1, 1, kernel_height, kernel_width)
    output = F.conv2d(output, weight)
    output = output.reshape(
        -1,
        minor,
        input_height * up_y + pad_y0 + pad_y1 - kernel_height + 1,
        input_width * up_x + pad_x0 + pad_x1 - kernel_width + 1,
    )
    output = output.permute(0, 2, 3, 1)
    output = output[:, ::down_y, ::down_x, :]
    output_height = (input_height * up_y + pad_y0 + pad_y1 - kernel_height) // down_y + 1
    output_width = (input_width * up_x + pad_x0 + pad_x1 - kernel_width) // down_x + 1
    return output.view(-1, channel, output_height, output_width)
