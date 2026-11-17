"""Integer remainder, fmod and floor division with negative operands.

Vulkan leaves OpSRem and OpSMod undefined when either operand is negative, and NVIDIA's driver returns
the unsigned remainder there (-6 % 7 gave 5): integer fmod, remainder, floor_divide and
div(rounding_mode="floor") were wrong for about half of all negative inputs, and so were the ops built
on them (take_along_dim with negative indices, integer index_reduce mean).
"""
import pytest
import torch

import torchvulkan  # noqa: F401

DEVICE = "vulkan"
INT_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
OPS = {
    "fmod": torch.fmod,
    "remainder": torch.remainder,
    "floor_divide": torch.floor_divide,
    "div_floor": lambda a, b: torch.div(a, b, rounding_mode="floor"),
    "div_trunc": lambda a, b: torch.div(a, b, rounding_mode="trunc"),
}


def operands(dtype):
    # every sign combination, divisors that do and don't divide, and the type's extremes
    info = torch.iinfo(dtype)
    a = torch.arange(-40, 41, dtype=torch.int64).repeat_interleave(15)
    b = torch.tensor([-9, -7, -5, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 9, 11]).repeat(81)
    extremes_a = torch.tensor([info.min + 1, info.max, info.min + 1, info.max, info.min, info.min])
    extremes_b = torch.tensor([7, -7, -7, 7, 3, 2])
    return torch.cat([a, extremes_a]).to(dtype), torch.cat([b, extremes_b]).to(dtype)


@pytest.mark.parametrize("dtype", INT_DTYPES, ids=str)
@pytest.mark.parametrize("op", OPS, ids=str)
def test_tensor_operands_match_cpu(op, dtype):
    a, b = operands(dtype)
    torch.testing.assert_close(OPS[op](a.to(DEVICE), b.to(DEVICE)).cpu(), OPS[op](a, b))


@pytest.mark.parametrize("dtype", INT_DTYPES, ids=str)
@pytest.mark.parametrize("op", OPS, ids=str)
@pytest.mark.parametrize("scalar", [7, -7, 3], ids=str)
def test_scalar_divisor_matches_cpu(op, dtype, scalar):
    a, _ = operands(dtype)
    torch.testing.assert_close(OPS[op](a.to(DEVICE), scalar).cpu(), OPS[op](a, scalar))


def test_take_along_dim_negative_indices():
    x = torch.randn(5, 7)
    idx = torch.randint(-7, 7, (5, 7))
    torch.testing.assert_close(torch.take_along_dim(x.to(DEVICE), idx.to(DEVICE), 1).cpu(), torch.take_along_dim(x, idx, 1))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=str)
def test_index_reduce_mean_rounds_down(dtype):
    # integer means of negative sums round toward -inf, as on the CPU
    torch.manual_seed(0)
    x = torch.randint(-9, 10, (5, 6), dtype=dtype)
    src = torch.randint(-9, 10, (5, 10), dtype=dtype)
    index = torch.tensor([0, 1, 1, 0, 1, 4, 5, 5, 2, 0])
    for include_self in (True, False):
        out = x.to(DEVICE).index_reduce(1, index.to(DEVICE), src.to(DEVICE), "mean", include_self=include_self).cpu()
        torch.testing.assert_close(out, x.index_reduce(1, index, src, "mean", include_self=include_self))
