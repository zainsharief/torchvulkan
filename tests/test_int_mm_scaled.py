import pytest
import torch
import torchvulkan

DEVICE = "vulkan"


def reference(a, b, scale_a, scale_b, bias, out_dtype):
    acc = (a.cpu().long() @ b.cpu().long()).double()
    out = acc * scale_a.cpu().double().reshape(-1, 1) * scale_b.cpu().double().reshape(1, -1)
    if bias is not None:
        out = out + bias.cpu().double().reshape(1, -1)
    return out


def make(m, k, n, scale_a_numel, scale_b_numel, with_bias, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(-128, 128, (m, k), dtype=torch.int8, generator=g)
    b = torch.randint(-128, 128, (k, n), dtype=torch.int8, generator=g)
    scale_a = torch.rand(scale_a_numel, 1, generator=g) * 1e-3 + 1e-4
    scale_b = torch.rand(1, scale_b_numel, generator=g) * 1e-3 + 1e-4
    bias = torch.randn(n, generator=g) if with_bias else None
    return a, b, scale_a, scale_b, bias


SHAPES = [
    (64, 64, 64),        # aligned
    (37, 96, 53),        # ragged M and N
    (256, 1024, 512),    # several K steps
    (4404, 256, 640),    # a token count that is not a multiple of any tile
    (1, 128, 96),        # one row
]


@pytest.mark.parametrize("m,k,n", SHAPES)
@pytest.mark.parametrize("per_row,per_col", [(True, True), (False, True), (True, False), (False, False)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_matches_int64_reference(m, k, n, per_row, per_col, with_bias):
    a, b, sa, sb, bias = make(m, k, n, m if per_row else 1, n if per_col else 1, with_bias)
    args = [t.to(DEVICE) if t is not None else None for t in (a, b, sa, sb, bias)]
    got = torchvulkan.int_mm_scaled(*args)
    assert got.device.type == DEVICE and got.dtype == torch.float32 and got.shape == (m, n)
    want = reference(a, b, sa, sb, bias, torch.float32)
    torch.testing.assert_close(got.cpu().double(), want, rtol=2e-6, atol=1e-5)


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_low_precision_output(out_dtype):
    a, b, sa, sb, bias = make(128, 512, 192, 128, 192, True, seed=1)
    got = torchvulkan.int_mm_scaled(a.to(DEVICE), b.to(DEVICE), sa.to(DEVICE), sb.to(DEVICE), bias.to(DEVICE), out_dtype)
    assert got.dtype == out_dtype
    # the float32 epilogue rounded once to the output type: within one ulp of the exact result
    want = reference(a, b, sa, sb, bias, torch.float32)
    ulp = 2.0 ** -10 if out_dtype == torch.float16 else 2.0 ** -7
    torch.testing.assert_close(got.cpu().double(), want, rtol=ulp, atol=1e-3)


def test_transposed_weight_view():
    # the usual linear layout: x [M, K] times W [N, K] transposed, without a copy
    a, b, sa, sb, bias = make(96, 256, 160, 96, 160, True, seed=2)
    w = b.t().contiguous()
    got = torchvulkan.int_mm_scaled(a.to(DEVICE), w.to(DEVICE).t(), sa.to(DEVICE), sb.to(DEVICE), bias.to(DEVICE))
    torch.testing.assert_close(got.cpu().double(), reference(a, b, sa, sb, bias, torch.float32), rtol=2e-6, atol=1e-5)


def test_matches_cpu_reference_op():
    a, b, sa, sb, bias = make(80, 192, 112, 80, 112, True, seed=3)
    cpu = torch.ops.torchvulkan.int_mm_scaled(a, b, sa, sb, bias, None)
    got = torch.ops.torchvulkan.int_mm_scaled(a.to(DEVICE), b.to(DEVICE), sa.to(DEVICE), sb.to(DEVICE), bias.to(DEVICE), None)
    torch.testing.assert_close(got.cpu(), cpu, rtol=2e-6, atol=1e-5)


def test_rejects_bad_arguments():
    a = torch.zeros(8, 16, dtype=torch.int8, device=DEVICE)
    b = torch.zeros(16, 4, dtype=torch.int8, device=DEVICE)
    s = torch.ones(1, device=DEVICE)
    with pytest.raises(RuntimeError, match="scale_a"):
        torchvulkan.int_mm_scaled(a, b, torch.ones(3, device=DEVICE), s)
    with pytest.raises(RuntimeError, match="float32 scales"):
        torchvulkan.int_mm_scaled(a, b, s.half(), s)
    with pytest.raises(RuntimeError, match="int8 inputs"):
        torchvulkan.int_mm_scaled(a.float(), b, s, s)
    with pytest.raises(RuntimeError, match="out_dtype"):
        torchvulkan.int_mm_scaled(a, b, s, s, out_dtype=torch.int32)


def quantize_reference(x, qmax):
    xf = x.cpu().float()
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    q = (xf * (qmax / amax)).round().clamp(-qmax, qmax).to(torch.int8)
    return q, amax / qmax


def check_quantized(q, scale, want_q, want_scale):
    # Vulkan divides within 2.5 ulp: the scale may differ in its last bits, and a code whose x * qmax / amax lands on
    # a .5 boundary may round the other way
    torch.testing.assert_close(scale.cpu(), want_scale, rtol=3e-7, atol=0)
    diff = (q.cpu().int() - want_q.int()).abs()
    assert diff.max().item() <= 1
    assert diff.count_nonzero().item() <= max(1, want_q.numel() // 1000)


@pytest.mark.parametrize("rows,cols", [(1, 7), (37, 300), (4150, 3072), (64, 8192), (5, 65536)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("qmax", [127.0, 7.0])
def test_quantize_per_row_matches_reference(rows, cols, dtype, qmax):
    g = torch.Generator().manual_seed(rows * cols)
    x = (torch.randn(rows, cols, generator=g) * torch.rand(rows, 1, generator=g) * 10).to(dtype)
    q, scale = torchvulkan.quantize_per_row(x.to(DEVICE), qmax)
    want_q, want_scale = quantize_reference(x, qmax)
    assert q.dtype == torch.int8 and scale.dtype == torch.float32 and scale.shape == (rows, 1)
    check_quantized(q, scale, want_q, want_scale)


def test_quantize_per_row_zero_rows_and_strided_input():
    x = torch.zeros(4, 64)
    x[2] = torch.linspace(-3, 3, 64)
    base = torch.randn(4, 128)
    base[:, ::2] = x
    view = base.to(DEVICE)[:, ::2]  # a strided view: copied to contiguous first
    q, scale = torchvulkan.quantize_per_row(view)
    want_q, want_scale = quantize_reference(x, 127.0)
    check_quantized(q, scale, want_q, want_scale)


def test_quantize_then_int_mm_scaled_matches_float_linear():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(300, 1024, generator=g)
    w = torch.randint(-127, 128, (512, 1024), dtype=torch.int8, generator=g)
    w_scale = torch.rand(1, 512, generator=g) * 1e-2
    q, s = torchvulkan.quantize_per_row(x.to(DEVICE))
    got = torchvulkan.int_mm_scaled(q, w.to(DEVICE).t(), s, w_scale.to(DEVICE))
    want = x.double() @ (w.double() * w_scale.double().t()).t()
    rel = ((got.cpu().double() - want).norm() / want.norm()).item()
    assert rel < 1e-2  # int8 activation quantization error
