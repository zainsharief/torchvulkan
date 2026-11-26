"""bfloat16 copies and matmuls on the device: bit-exact conversions against the CPU, matmuls against a float32 CPU
reference, and no CPU fallback under strict mode. On devices with shaderBFloat16 these run the native path; on
devices without it (RADV on RDNA3) they run the integer-op emulation."""
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

import torchvulkan  # noqa: F401

DEVICE = 'vulkan'


def all_bf16_bits():
    return torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)


def test_bfloat16_to_float32_is_exact_for_every_bit_pattern():
    x = all_bf16_bits()
    out = x.to(DEVICE).to(torch.float32).cpu()
    assert torch.equal(out.view(torch.int32), x.to(torch.float32).view(torch.int32))


def float32_cases():
    g = torch.Generator().manual_seed(0)
    random_bits = torch.randint(-2**31, 2**31 - 1, (1 << 20,), dtype=torch.int64, generator=g).to(torch.int32)
    # ties: the low 16 bits exactly 0x8000, both parities of bit 16
    ties = (torch.arange(0, 1 << 16, dtype=torch.int32) << 16) | 0x8000
    special = torch.tensor([0.0, -0.0, float('inf'), float('-inf'), float('nan'), 3.3895313892515355e38, -3.3895313892515355e38,
                            1e-40, -1e-40, 1.17549435e-38, 65504.0, 1.0, -1.0, 0.1], dtype=torch.float32)
    return torch.cat([random_bits.view(torch.float32), ties.view(torch.float32), special])


def test_float32_to_bfloat16_rounds_like_the_cpu():
    x = float32_cases()
    out = x.to(DEVICE).to(torch.bfloat16).cpu()
    assert torch.equal(out.view(torch.int16), x.to(torch.bfloat16).view(torch.int16))


@pytest.mark.parametrize('src', [torch.float16, torch.float32])
def test_bfloat16_round_trips_through_other_dtypes(src):
    x = float32_cases()[: 1 << 16]
    x = x[torch.isfinite(x)].to(src)
    got = x.to(DEVICE).to(torch.bfloat16)
    assert torch.equal(got.cpu().view(torch.int16), x.to(torch.bfloat16).view(torch.int16))
    back = got.to(src).cpu()
    assert torch.equal(back, x.to(torch.bfloat16).to(src))


def test_bfloat16_copy_into_a_strided_view():
    x = torch.randn(64, 96)
    dst = torch.zeros(96, 64, dtype=torch.bfloat16, device=DEVICE).t()
    dst.copy_(x.to(DEVICE))
    assert torch.equal(dst.cpu().view(torch.int16), x.to(torch.bfloat16).view(torch.int16))


def reference(fn, *args):
    return fn(*[a.float() if a.is_floating_point() else a for a in args])


@pytest.mark.parametrize('m,k,n', [(1, 1, 1), (17, 33, 9), (256, 512, 128), (300, 1000, 260)])
def test_bfloat16_mm_linear_addmm(m, k, n):
    torch.manual_seed(0)
    a = torch.randn(m, k).bfloat16(); b = torch.randn(k, n).bfloat16(); w = torch.randn(n, k).bfloat16(); bias = torch.randn(n).bfloat16()
    ad, bd, wd, biasd = (t.to(DEVICE) for t in (a, b, w, bias))
    tol = dict(rtol=1.6e-2, atol=1e-2 * k ** 0.5)
    out = ad @ bd
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.cpu().float(), reference(torch.mm, a, b), **tol)
    torch.testing.assert_close(F.linear(ad, wd, biasd).cpu().float(), reference(F.linear, a, w, bias), **tol)
    torch.testing.assert_close(torch.addmm(biasd, ad, bd, beta=0.5, alpha=2.0).cpu().float(),
                               torch.addmm(bias.float(), a.float(), b.float(), beta=0.5, alpha=2.0), **tol)


def test_bfloat16_bmm():
    torch.manual_seed(0)
    a = torch.randn(3, 40, 70).bfloat16(); b = torch.randn(3, 70, 50).bfloat16()
    out = torch.bmm(a.to(DEVICE), b.to(DEVICE))
    torch.testing.assert_close(out.cpu().float(), torch.bmm(a.float(), b.float()), rtol=1.6e-2, atol=0.1)


def test_bfloat16_matmul_and_copy_stay_on_the_device():
    code = (
        "import torch, torchvulkan\n"
        "a = torch.randn(64, 32).bfloat16().to('vulkan'); b = torch.randn(32, 48).bfloat16().to('vulkan')\n"
        "c = (a @ b).float(); d = torch.randn(8).to('vulkan').bfloat16(); torchvulkan.synchronize(); print('ok')\n"
    )
    env = dict(os.environ, TORCHVULKAN_STRICT='1')
    r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and 'ok' in r.stdout, r.stderr[-2000:]


def test_bfloat16_copy_broadcasts_and_writes_contiguous_destinations():
    x = torch.randn(1, 96)
    dst = torch.zeros(64, 96, dtype=torch.bfloat16, device=DEVICE)
    dst.copy_(x.to(DEVICE))   # float32 -> bfloat16, broadcast
    assert torch.equal(dst.cpu().view(torch.int16), x.expand(64, 96).to(torch.bfloat16).view(torch.int16))
    wide = torch.zeros(64, 96, device=DEVICE)
    wide.copy_(x.to(torch.bfloat16).to(DEVICE))   # bfloat16 -> float32, broadcast
    assert torch.equal(wide.cpu(), x.expand(64, 96).to(torch.bfloat16).float())
    y = torch.randn(33, 17).bfloat16()
    out = torch.empty(33, 17, device=DEVICE)
    out.copy_(y.to(DEVICE))   # contiguous: written in place
    assert torch.equal(out.cpu(), y.float())
