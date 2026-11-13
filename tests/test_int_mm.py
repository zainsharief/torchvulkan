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


# 'default' takes the cooperative-matrix kernel where the device has an int8 -> int32
# configuration; 'scalar' forces the scalar kernel, so both stay covered
@pytest.fixture(params=['default', 'scalar'])
def kernel_path(request, monkeypatch):
    if request.param == 'scalar':
        monkeypatch.setenv('TORCHVULKAN_INT_MM_SCALAR', '1')
    else:
        monkeypatch.delenv('TORCHVULKAN_INT_MM_SCALAR', raising=False)
    return request.param


def check(a, b):
    expected = torch._int_mm(a, b)
    result = torch._int_mm(a.to(DEVICE), b.to(DEVICE))
    assert result.dtype == torch.int32
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize('m,k,n', SHAPES)
@pytest.mark.parametrize('layout', ['contiguous', 'transposed_b', 'transposed_a'])
def test_int_mm_matches_cpu(m, k, n, layout, kernel_path):
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


# whole tiles (the result is stored straight into the output), K tails of every size
# below the 16-deep fragment, and a K that is not a multiple of the staged K step
@pytest.mark.parametrize('m,k,n', [
    (128, 256, 128),
    (256, 512, 512),
    (128, 17, 128),
    (64, 1, 64),
    (32, 15, 48),
    (96, 4095, 80),
    (5, 4096, 3),
])
@pytest.mark.parametrize('layout', ['contiguous', 'transposed_b'])
def test_int_mm_tile_and_k_edges(m, k, n, layout, kernel_path):
    g = torch.Generator().manual_seed(m * 31 + k * 7 + n)
    a = ints(m, k, generator=g)
    b = ints(n, k, generator=g).t() if layout == 'transposed_b' else ints(k, n, generator=g)
    check(a, b)


def test_int_mm_strided_operands(kernel_path):
    g = torch.Generator().manual_seed(11)
    big_a = ints(80, 200, generator=g)
    big_b = ints(200, 140, generator=g)
    check(big_a[3:67, 5:133], big_b[1:129, 7:71])        # row-major views with padded rows, odd offsets
    check(big_a[::2, ::2], big_b[::2, ::2])               # no unit stride at all (scalar fallback)
    check(big_a[:64, :128], big_b[:100, :128].t())        # transposed view of a sliced weight


def test_int_mm_out_at_unaligned_offset(kernel_path):
    g = torch.Generator().manual_seed(5)
    a, b = ints(64, 64, generator=g), ints(64, 64, generator=g)
    expected = torch._int_mm(a, b)
    storage = torch.zeros(64 * 64 + 1, dtype=torch.int32, device=DEVICE)
    out = storage[1:].view(64, 64)
    torch._int_mm(a.to(DEVICE), b.to(DEVICE), out=out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert storage[0].item() == 0


def test_int_mm_uses_cooperative_matrices_when_available(monkeypatch, capfd):
    monkeypatch.delenv('TORCHVULKAN_INT_MM_SCALAR', raising=False)
    monkeypatch.setenv('TORCHVULKAN_INT_MM_DEBUG', '1')
    a = torch.ones((64, 64), dtype=torch.int8, device=DEVICE)
    torch._int_mm(a, a)
    import torchvulkan as tv
    tv.synchronize()
    err = capfd.readouterr().err
    if '_int_mm coop' not in err:
        pytest.skip('the device has no int8 x int8 -> int32 cooperative-matrix configuration')


def test_int_mm_extreme_values_accumulate_exactly(kernel_path):
    k = 2048
    a = torch.full((8, k), -128, dtype=torch.int8)
    b = torch.full((k, 8), -128, dtype=torch.int8)
    result = torch._int_mm(a.to(DEVICE), b.to(DEVICE)).cpu()
    assert torch.all(result == 128 * 128 * k)


def test_int_mm_out_variant_and_strided_out(kernel_path):
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
