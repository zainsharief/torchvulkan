"""A process that used the vulkan device exits cleanly.

The context is torn down at exit (Linux and macOS), after Python has finalised and before shared
libraries unload. Without that, NVIDIA's Linux driver crashed every such process at exit, because its
worker threads outlived the unloaded driver. Each script runs in a fresh interpreter.
"""
import subprocess
import sys
import tempfile
import textwrap

import pytest


def run_script(source):
    # `python -c` puts the working directory first on sys.path; run from the temp directory so the child
    # imports the installed torchvulkan, not the source tree's package (which has no compiled _C)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(source)], capture_output=True, text=True, timeout=300,
                          cwd=tempfile.gettempdir())


SCRIPTS = {
    "tensors freed": (0, """
        import torch, torchvulkan
        x = torch.ones(1024, device="vulkan")
        print((x * 2).sum().item())
        del x
    """),
    "tensors alive at exit": (0, """
        import torch, torchvulkan
        keep = [torch.randn(256, 256, device=torch.device("vulkan", i)) @ torch.randn(256, 256, device=torch.device("vulkan", i))
                for i in range(torchvulkan.device_count())]
        print(len(keep))
    """),
    "cached buffers": (0, """
        import torch, torchvulkan
        for n in (1, 7, 100, 4096, 65536, 1 << 20):
            y = torch.empty(n, device="vulkan")
            y.fill_(1.0)
            del y
        print("done")
    """),
    "uncaught exception": (1, """
        import torch, torchvulkan
        x = torch.ones(4, device="vulkan")
        raise RuntimeError("deliberate")
    """),
}


@pytest.mark.parametrize("name", list(SCRIPTS))
def test_exit_code(name):
    expected, source = SCRIPTS[name]
    result = run_script(source)
    assert result.returncode == expected, f"exit code {result.returncode}\n{result.stderr[-2000:]}"
