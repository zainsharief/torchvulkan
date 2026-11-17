"""tanh (and gelu with approximate='tanh', which uses it) for large inputs, and atanh at |x| >= 1.

Vulkan defines tanh's precision through (e^x - e^-x) / (e^x + e^-x), so a driver may overflow to
inf / inf = NaN for large |x| (AMD's Windows driver did from about 89); the result must saturate to
+-1 as on the CPU. Diffusion models feed tanh-approximated GELU activations far beyond that range.
"""
import pytest
import torch
import torch.nn.functional as F

import torchvulkan

DEVICE = "vulkan"
DTYPES = [torch.float32, torch.float16, torch.float64] + ([torch.bfloat16] if torchvulkan.is_bf16_supported() else [])


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_tanh_saturates(dtype):
    x = torch.tensor([0.0, 0.5, -3.0, 9.0, 15.0, 16.0, 44.0, 45.0, 89.0, 100.0, 1e4, -50.0, -1e4,
                      float("inf"), -float("inf")]).to(dtype)
    if dtype == torch.float16:
        x = x.clamp(-6e4, 6e4)
    out = torch.tanh(x.to(DEVICE)).cpu()
    torch.testing.assert_close(out, torch.tanh(x), rtol=1e-3, atol=1e-3)


def test_tanh_propagates_nan():
    out = torch.tanh(torch.tensor([float("nan"), 1.0], device=DEVICE)).cpu()
    assert out[0].isnan() and out[1] == torch.tanh(torch.tensor(1.0))


@pytest.mark.parametrize("dtype", DTYPES + [torch.int32, torch.int64], ids=str)
def test_atanh_domain_edges(dtype):
    # Vulkan leaves atanh undefined for |x| >= 1 (AMD's driver returned +-1.7e38 at +-1, not +-inf)
    x = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]).to(dtype)
    torch.testing.assert_close(torch.atanh(x.to(DEVICE)).cpu(), torch.atanh(x), equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=str)
def test_gelu_tanh_large_inputs(dtype):
    x = (torch.randn(4096) * 300).to(dtype)
    out = F.gelu(x.to(DEVICE), approximate="tanh").cpu()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), F.gelu(x.float(), approximate="tanh"), rtol=2e-3, atol=2e-3)
