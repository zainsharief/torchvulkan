"""Scalars passed to shaders (fill values, alpha, beta, scalar operands) keep their value in 8- and 16-bit dtypes.

Those scalars travel to the shader widened to 32 bits (T.PushScalar in core/dtype.slang), so a device
without storagePushConstant8/16 can run the kernels. These check the round trip: the extremes of each
dtype, negative values (sign extension), and values that only exist after rounding to the dtype.
"""
import pytest
import torch
import torchvulkan  # noqa: F401  (registers the vulkan device)

SCALARS = {
    torch.int8: [-128, -1, 0, 127],
    torch.uint8: [0, 1, 255],
    torch.int16: [-32768, -1, 32767],
    torch.float16: [0.1, -65504.0, 6e-8, 3.14159],     # 6e-8 is subnormal in float16
    torch.bfloat16: [0.1, -3.0e38, 1e-30, 3.14159],
}
CASES = [(dtype, value) for dtype, values in SCALARS.items() for value in values]


def check(fn, dtype):
    base = (torch.arange(-8, 9) % 5).to(dtype)
    expected = fn(base)
    actual = fn(base.to("vulkan")).cpu()
    torch.testing.assert_close(actual, expected, equal_nan=True)


@pytest.mark.parametrize("dtype,value", CASES, ids=str)
def test_fill(dtype, value):
    check(lambda t: torch.full_like(t, value), dtype)


@pytest.mark.parametrize("dtype,value", CASES, ids=str)
def test_binary_scalar(dtype, value):
    if dtype.is_floating_point:
        check(lambda t: t * value, dtype)
    else:
        check(lambda t: torch.bitwise_and(t, value), dtype)


# Vulkan lets a device flush float16 subnormals to zero in arithmetic (Intel's Gen9 does), so comparing
# against one is device-dependent; test_fill still checks that a subnormal arrives intact.
NORMAL_CASES = [(dtype, value) for dtype, value in CASES if not (dtype == torch.float16 and abs(value) < 6.2e-5 and value != 0)]


@pytest.mark.parametrize("dtype,value", NORMAL_CASES, ids=str)
def test_compare_scalar(dtype, value):
    check(lambda t: t >= value, dtype)


@pytest.mark.parametrize("dtype", list(SCALARS), ids=str)
def test_add_alpha(dtype):
    alpha = 0.5 if dtype.is_floating_point else -3
    check(lambda t: torch.add(t, t, alpha=alpha), dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8, torch.int16], ids=str)
def test_addmm_alpha_beta(dtype):
    alpha, beta = (0.5, 0.25) if dtype.is_floating_point else (2, -1)
    m = (torch.arange(16).reshape(4, 4) % 3).to(dtype)
    torch.testing.assert_close(torch.addmm(m.to("vulkan"), m.to("vulkan"), m.to("vulkan"), alpha=alpha, beta=beta).cpu(),
                               torch.addmm(m, m, m, alpha=alpha, beta=beta))


def test_bool_fill():
    check(lambda t: torch.full_like(t, True), torch.bool)
    check(lambda t: torch.full_like(t, False), torch.bool)
