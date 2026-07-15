import pytest
import torch

from RSB.backbone.ncsnpp.ncsnpp_utils.op.backend import operator_backend, use_pytorch_native
from RSB.backbone.ncsnpp.ncsnpp_utils.op.fused_act import fused_leaky_relu
from RSB.backbone.ncsnpp.ncsnpp_utils.op.upfirdn2d import upfirdn2d


def exercise_native_ops(device: torch.device) -> None:
    use_pytorch_native()
    inputs = torch.randn(2, 3, 4, 4, device=device, requires_grad=True)
    bias = torch.randn(3, device=device, requires_grad=True)
    kernel = torch.tensor([[1.0, 2.0], [2.0, 1.0]], device=device)
    kernel /= kernel.sum()

    activated = fused_leaky_relu(inputs, bias, negative_slope=0.1)
    filtered = upfirdn2d(activated, kernel, up=2, pad=(1, 1))
    filtered.square().mean().backward()

    assert filtered.shape == (2, 3, 9, 9)
    assert torch.isfinite(filtered).all()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert bias.grad is not None
    assert torch.isfinite(bias.grad).all()
    assert operator_backend() == "pytorch_native"


def test_native_ncsnpp_ops_support_cpu_autograd() -> None:
    exercise_native_ops(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_ncsnpp_ops_support_cuda_autograd() -> None:
    exercise_native_ops(torch.device("cuda:0"))
