"""Tensors on a Vulkan device other than the current one (vulkan:1 while vulkan:0 is current), as
when a model is split across GPUs. Every op must allocate on its tensors' own device.

Needs two Vulkan devices; skipped otherwise.
"""
import pytest
import torch

import torchvulkan

pytestmark = pytest.mark.skipif(torchvulkan.device_count() < 2, reason="needs two Vulkan devices")


@pytest.fixture(params=[0, 1], ids=["vulkan:0", "vulkan:1"])
def device(request):
    torchvulkan.set_device(0)  # keep vulkan:0 current, so vulkan:1 is the non-current device
    return torch.device("vulkan", request.param)


def test_integer_arange_used_by_kernels(device):
    # arange is computed on the CPU and copied into an out= tensor that starts empty; the GPU
    # kernels below then read it on its own device
    a = torch.arange(0, 128, 2, device=device)
    assert a.device == device
    torch.testing.assert_close(a.float().cpu(), torch.arange(0, 128, 2).float())
    torch.testing.assert_close((a * 3).cpu(), torch.arange(0, 128, 2) * 3)


def test_out_variant_starting_empty(device):
    x = torch.randn(64, 33)
    y = torch.randn(64, 33)
    out = torch.empty(0, device=device)
    torch.add(x.to(device), y.to(device), out=out)
    assert out.device == device
    torch.testing.assert_close(out.cpu(), x + y)
    torch.testing.assert_close((out * 2).cpu(), (x + y) * 2)  # read by a kernel on that device


def test_resize_keeps_device_and_data(device):
    t = torch.arange(10, dtype=torch.float32, device=device)
    t.resize_(1000)
    t[10:] = 7.0
    assert t.device == device
    expected = torch.cat([torch.arange(10, dtype=torch.float32), torch.full((990,), 7.0)])
    torch.testing.assert_close((t + 0).cpu(), expected)
