"""float16 results on the vulkan device, compared exactly with the CPU.

Vulkan leaves the rounding mode of float conversions to the driver (AMD's rounds float32 -> float16
toward zero), and lets division, square roots and transcendental functions be off by a few ulp in
float16. torchvulkan rounds conversions to float16 to nearest even itself, and computes float16
math in float32 and rounds once, as PyTorch does on the CPU and CUDA.
"""
import pytest
import torch

import torchvulkan  # noqa: F401  registers the vulkan device

DEVICE = "vulkan"
N = 1 << 18


def data(seed, scale=10.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N, generator=g) * scale


def assert_same_as_cpu(out, ref):
    out = out.cpu()
    differ = (out.float() != ref.float()) & ~(out.isnan() & ref.isnan())
    assert int(differ.sum()) == 0, (
        f"{int(differ.sum())} of {ref.numel()} values differ from the CPU, e.g. "
        f"{out[differ][:3].tolist()} vs {ref[differ][:3].tolist()}")


@pytest.mark.parametrize("src", [torch.float32, torch.float64], ids=str)
def test_cast_to_float16_rounds_to_nearest_even(src):
    x = data(0).to(src)
    # also values exactly halfway between two float16 values, which round to the even one, and
    # values past float16's range, subnormals and signed zeros
    halfway = torch.tensor([1 + 2**-11, 1 + 3 * 2**-11, 2048 + 1, 2048 + 3, -(1 + 2**-11)], dtype=src)
    edge = torch.tensor([65504, 65519, 65520, 1e6, 6e-8, 3e-8, 2**-25, -0.0, 0.0], dtype=src)
    for values in (x, halfway, edge):
        assert_same_as_cpu(values.to(DEVICE).half(), values.half())


def test_cast_integers_to_float16():
    x = torch.randint(-70000, 70000, (N,), dtype=torch.int64)
    assert_same_as_cpu(x.to(DEVICE).half(), x.half())


def test_float16_sum_rounds_once():
    x = data(1, scale=3.0).half().reshape(256, -1)
    # summed in float32 and rounded to float16 once; with these sizes the float32 sums agree
    # with the CPU's, so the rounding decides
    assert_same_as_cpu(x.to(DEVICE).sum(1), x.sum(1))


@pytest.mark.parametrize("op", ["div", "sqrt", "reciprocal", "rsqrt"])
def test_float16_math_matches_cpu(op):
    a = data(2).half()
    b = (data(3).abs() + 0.1).half()
    fns = {
        "div": lambda p, q: p / q,
        "sqrt": lambda p, q: q.sqrt(),
        "reciprocal": lambda p, q: q.reciprocal(),
        "rsqrt": lambda p, q: q.rsqrt(),
    }
    fn = fns[op]
    assert_same_as_cpu(fn(a.to(DEVICE), b.to(DEVICE)), fn(a, b))


@pytest.mark.parametrize("op", ["exp", "log", "cos", "tanh", "sigmoid"])
def test_float16_transcendentals_within_one_ulp(op):
    # computed in float32, whose own functions may be an ulp off in float32: after rounding to
    # float16 at most one float16 step. (Not sin: Vulkan bounds only its absolute error, so near
    # zero NVIDIA's float32 sin is relatively inaccurate, e.g. sin(1.7e-5) is 1.2e-7 low.)
    x = (data(4, scale=2.0)).half()
    if op == "log":
        x = x.abs() + 0.01
    fn = getattr(torch, op)
    out = fn(x.to(DEVICE)).cpu().float()
    ref = fn(x).float()
    ulp = torch.finfo(torch.float16).eps * ref.abs().clamp(min=torch.finfo(torch.float16).tiny)
    bad = (out - ref).abs() > ulp * 1.001
    assert not bad.any(), (f"{int(bad.sum())} values off by more than an ulp, e.g. x={x[bad][:3].tolist()}: "
                           f"{out[bad][:3].tolist()} vs {ref[bad][:3].tolist()}")
