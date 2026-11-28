import os
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn.functional as F
import torchvulkan

DEVICE = "vulkan"

# one bfloat16 convolution per process, against a private cache directory so the user's cache is never touched
SCRIPT = textwrap.dedent("""
    import sys, torch, torchvulkan, torch.nn.functional as F
    mode = sys.argv[1]
    x = torch.randn(1, 32, 64, 64, device="vulkan", dtype=torch.bfloat16)
    w = torch.randn(32, 32, 3, 3, device="vulkan", dtype=torch.bfloat16)
    if mode == "retune":
        with torchvulkan.tuning.retune():
            F.conv2d(x, w, padding=1)
    else:
        F.conv2d(x, w, padding=1)
    torchvulkan.synchronize()
    if mode == "clear":
        torchvulkan.tuning.clear(persistent=True)
    print("entries", torchvulkan.tuning.entries())
    print("file", torchvulkan.tuning.cache_file())
""")


def run(mode, cache_dir):
    env = dict(os.environ, TORCHVULKAN_CACHE_DIR=str(cache_dir), TORCHVULKAN_TUNE_LOG="1")
    env.pop("TORCHVULKAN_RETUNE", None)
    p = subprocess.run([sys.executable, "-c", SCRIPT, mode], env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout, p.stderr


def path_lines(stderr):
    return [l for l in stderr.splitlines() if l.startswith("torchvulkan tune: path ") or l.startswith("torchvulkan tune: chose path")]


def test_paths_are_timed_once_stored_retuned_and_cleared(tmp_path):
    out, err = run("plain", tmp_path)
    first = path_lines(err)
    if not first:
        pytest.skip("one path only on this device (no choice to time)")
    assert any(l.startswith("torchvulkan tune: chose path") for l in first)
    tuning_file = [l.split(" ", 1)[1] for l in out.splitlines() if l.startswith("file ")][0]
    assert tuning_file and os.path.getsize(tuning_file) > 0

    # a second process reuses the stored choice
    out, err = run("plain", tmp_path)
    assert path_lines(err) == []
    assert int(out.split("entries ")[1].split()[0]) > 0

    # retune times it again
    out, err = run("retune", tmp_path)
    assert any(l.startswith("torchvulkan tune: chose path") for l in path_lines(err))

    # clear(persistent=True) forgets and empties the file
    out, err = run("clear", tmp_path)
    assert int(out.split("entries ")[1].split()[0]) == 0
    assert os.path.getsize(tuning_file) == 0


CONVS = [
    # name, input, weight, kwargs, transposed
    ("3x3", (2, 16, 33, 31), (24, 16, 3, 3), dict(padding=1), False),
    ("1x1", (1, 32, 40, 40), (48, 32, 1, 1), dict(), False),
    ("strided grouped", (1, 32, 37, 29), (32, 8, 3, 3), dict(stride=2, padding=1, groups=4), False),
    ("transposed", (1, 16, 30, 30), (16, 8, 4, 4), dict(stride=2, padding=1), True),
]


@pytest.mark.parametrize("name,xs,ws,kwargs,transposed", CONVS, ids=[c[0] for c in CONVS])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_convolution_forward_and_backward(name, xs, ws, kwargs, transposed, dtype):
    g = torch.Generator().manual_seed(len(name))
    x = torch.randn(*xs, generator=g).to(dtype)
    w = (torch.randn(*ws, generator=g) / (ws[1] * ws[2] * ws[3]) ** 0.5).to(dtype)
    b = torch.randn(ws[1] * kwargs.get("groups", 1) if transposed else ws[0], generator=g).to(dtype)
    conv = F.conv_transpose2d if transposed else F.conv2d

    xd, wd, bd = (t.to(DEVICE).requires_grad_() for t in (x, w, b))
    out = conv(xd, wd, bd, **kwargs)
    out.backward(torch.ones_like(out))

    xc, wc, bc = (t.float().requires_grad_() for t in (x, w, b))
    ref = conv(xc, wc, bc, **kwargs)
    ref.backward(torch.ones_like(ref))

    tol = {torch.bfloat16: 2e-2, torch.float16: 4e-3, torch.float32: 1e-4}[dtype]
    for got, want in ((out, ref), (xd.grad, xc.grad), (wd.grad, wc.grad), (bd.grad, bc.grad)):
        assert got.dtype == dtype
        torch.testing.assert_close(got.detach().cpu().float(), want.detach(), rtol=tol, atol=tol * want.abs().max().item())
