import os
import pytest
import torch
import torch.nn.functional as F
import torchvulkan  # noqa: F401

DEVICE = "vulkan"


@pytest.fixture(autouse=True)
def warp_kernel_only(monkeypatch):
    # the tuner picks the fastest kernel; pin the forward to attention_warp so these shapes exercise it
    monkeypatch.setenv("TORCHVULKAN_ATTENTION_KERNEL", "2")


def reference(q, k, v, **kwargs):
    out = F.scaled_dot_product_attention(q.cpu().double(), k.cpu().double(), v.cpu().double(),
                                         **{n: (t.cpu().double() if isinstance(t, torch.Tensor) and t.dtype != torch.bool else (t.cpu() if isinstance(t, torch.Tensor) else t))
                                            for n, t in kwargs.items()})
    return out


SHAPES = [
    # B, Hq, Hk, L, S, D, Dv
    (1, 4, 4, 128, 128, 64, 64),
    (1, 24, 24, 300, 333, 128, 128),     # ragged rows and keys
    (2, 3, 3, 17, 45, 80, 80),           # head size not a multiple of the tile
    (1, 8, 2, 96, 160, 64, 64),          # grouped-query attention
    (1, 2, 2, 1, 70, 128, 128),          # one query row
    (1, 24, 24, 1100, 1100, 128, 128),   # a DiT-like block
]


@pytest.mark.parametrize("B,Hq,Hk,L,S,D,Dv", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_matches_cpu(B, Hq, Hk, L, S, D, Dv, dtype):
    g = torch.Generator().manual_seed(L * S + D)
    q = torch.randn(B, Hq, L, D, generator=g)
    k = torch.randn(B, Hk, S, D, generator=g)
    v = torch.randn(B, Hk, S, Dv, generator=g)
    got = F.scaled_dot_product_attention(q.to(DEVICE, dtype), k.to(DEVICE, dtype), v.to(DEVICE, dtype), enable_gqa=Hq != Hk)
    want = reference(q.to(dtype), k.to(dtype), v.to(dtype), enable_gqa=Hq != Hk)
    tol = 2e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(got.cpu().double(), want, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_causal_and_mask(dtype):
    g = torch.Generator().manual_seed(5)
    q, k, v = (torch.randn(1, 4, 200, 64, generator=g) for _ in range(3))
    for kwargs in ({"is_causal": True},
                   {"attn_mask": torch.randn(1, 4, 200, 200, generator=g)},
                   {"attn_mask": torch.rand(1, 1, 200, 200, generator=g) > 0.3}):
        dev = {n: (t.to(DEVICE, dtype if t.dtype != torch.bool else torch.bool) if isinstance(t, torch.Tensor) else t) for n, t in kwargs.items()}
        got = F.scaled_dot_product_attention(q.to(DEVICE, dtype), k.to(DEVICE, dtype), v.to(DEVICE, dtype), **dev)
        ref_kwargs = {n: (t.to(dtype) if isinstance(t, torch.Tensor) and t.dtype != torch.bool else t) for n, t in kwargs.items()}
        want = reference(q.to(dtype), k.to(dtype), v.to(dtype), **ref_kwargs)
        tol = 2e-3 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(got.cpu().double(), want, rtol=tol, atol=tol)


def test_fully_masked_row_gives_zeros_like_cpu():
    q, k, v = (torch.randn(1, 1, 8, 64) for _ in range(3))
    mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    mask[0, 0, 3] = False
    got = F.scaled_dot_product_attention(q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), attn_mask=mask.to(DEVICE))
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(got.cpu(), want, rtol=1e-4, atol=1e-4, equal_nan=True)
