import pytest
import torch
import torchvulkan  # noqa: F401  registers the vulkan device

DEVICE = 'vulkan'

SHAPES = [
    (1, 32, 8),
    (17, 40, 24),
    (64, 64, 64),
    (65, 33, 129),
    (130, 257, 70),
    (300, 1024, 512),
]


def ints(*shape, generator):
    return torch.randint(-128, 128, shape, dtype=torch.int8, generator=generator)


@pytest.mark.parametrize('m,k,n', SHAPES)
@pytest.mark.parametrize('layout', ['contiguous', 'transposed_b', 'transposed_a'])
def test_int_mm_matches_cpu(m, k, n, layout):
    g = torch.Generator().manual_seed(m * 1000003 + k * 101 + n)
    a = ints(m, k, generator=g)
    b = ints(k, n, generator=g)
    if layout == 'transposed_b':  # how a Linear weight is passed: weight[N, K].t()
        b = ints(n, k, generator=g).t()
    elif layout == 'transposed_a':
        a = ints(k, m, generator=g).t()
    expected = torch._int_mm(a, b)
    result = torch._int_mm(a.to(DEVICE), b.to(DEVICE))
    assert result.dtype == torch.int32
    assert result.device.type == DEVICE
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


def test_int_mm_extreme_values_accumulate_exactly():
    k = 2048
    a = torch.full((8, k), -128, dtype=torch.int8)
    b = torch.full((k, 8), -128, dtype=torch.int8)
    result = torch._int_mm(a.to(DEVICE), b.to(DEVICE)).cpu()
    assert torch.all(result == 128 * 128 * k)


def test_int_mm_out_variant_and_strided_out():
    g = torch.Generator().manual_seed(7)
    a, b = ints(40, 64, generator=g), ints(64, 24, generator=g)
    expected = torch._int_mm(a, b)
    out = torch.empty((40, 24), dtype=torch.int32, device=DEVICE)
    torch._int_mm(a.to(DEVICE), b.to(DEVICE), out=out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    wide = torch.zeros((24, 40), dtype=torch.int32, device=DEVICE).t()  # non-row-major out
    torch._int_mm(a.to(DEVICE), b.to(DEVICE), out=wide)
    torch.testing.assert_close(wide.cpu(), expected, rtol=0, atol=0)


def test_int_mm_rejects_wrong_dtypes_and_shapes():
    a = torch.zeros((8, 16), dtype=torch.int8, device=DEVICE)
    with pytest.raises(RuntimeError):
        torch._int_mm(a.float(), a.t().float())
    with pytest.raises(RuntimeError):
        torch._int_mm(a, torch.zeros((8, 8), dtype=torch.int8, device=DEVICE))


def test_int_mm_has_a_vulkan_kernel():
    # without one, _int_mm goes through the CPU fallback (strict mode is read once at startup)
    for op in ('aten::_int_mm', 'aten::_int_mm.out'):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(op, 'PrivateUse1'), op
