"""Staging buffers: the host memory that copies between the CPU and the GPU go through.

Freed staging buffers are cached for reuse. The cache must stay small (it is host memory, and on
Windows it is charged to the process's commit, alongside the GPU memory itself), and an upload's
buffer (write-combined memory, slow for the CPU to read) must not be reused for a download.
"""
import time

import torch

import torchvulkan

DEVICE = "vulkan"
MB = 2**20


def staging_bytes():
    return torchvulkan.memory_stats()["staging_cached_bytes.all.current"]


def test_staging_cache_stays_capped():
    torchvulkan.empty_cache()
    # 1.5 GB of weights of 24 different sizes, uploaded and downloaded, as when a model is moved
    # to the GPU and back
    sizes = [(16 + 2 * i) * MB for i in range(24)]
    cpu = [torch.ones(s, dtype=torch.uint8) for s in sizes]
    gpu = [t.to(DEVICE) for t in cpu]
    back = [t.to("cpu") for t in gpu]
    torchvulkan.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(cpu, back))
    # upload and download pools are each capped at 256 MB by default
    assert staging_bytes() <= 2 * 256 * MB, f"{staging_bytes() / MB:.0f} MB of staging buffers cached"
    torchvulkan.empty_cache()
    assert staging_bytes() == 0


def download_rate(tensors, passes=3):
    """Downloads per second, the best of several passes (timing on a desktop is noisy)."""
    best = 0.0
    for _ in range(passes):
        torchvulkan.synchronize()
        t0 = time.perf_counter()
        for t in tensors:
            t.to("cpu")
        best = max(best, len(tensors) / (time.perf_counter() - t0))
    return best


def test_download_after_upload_is_not_slow():
    torchvulkan.empty_cache()
    n, size = 6, 32 * MB
    on_gpu = [torch.empty(size, dtype=torch.uint8, device=DEVICE) for _ in range(n)]
    clean = download_rate(on_gpu)
    # uploads of the same size leave their staging buffers in the cache; a download that reused
    # one read its result out of write-combined memory, ~50x slower, for as long as it stayed
    # cached. (The first pass afterwards is skipped: the upload buffers return to the cache at the
    # next flush, inside it, and trimming them costs time once.)
    for _ in range(n):
        torch.empty(size, dtype=torch.uint8).to(DEVICE)
    download_rate(on_gpu, passes=1)
    after_uploads = download_rate(on_gpu)
    assert after_uploads > clean / 8, f"downloads {clean / after_uploads:.0f}x slower after uploads"
