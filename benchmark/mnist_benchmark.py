"""Benchmark the MNIST forward + backward pass across every available backend.

Detects the compute backends on this machine (CPU, CUDA, MPS, and the
torchvulkan `vulkan` device), then times a combined forward + backward pass
(everything in ``float16``) through the MNIST network from ``examples/mnist.py``
at a range of model sizes. The hidden width is swept so the parameter count
grows from ~50K up into the hundreds of millions, and the per-backend timings
are plotted against the parameter count so you can see how each implementation
scales as ``n`` grows.

    python examples/benchmark.py                 # sensible defaults
    python examples/benchmark.py --iters 50      # more samples per point
    python examples/benchmark.py --out bench.png # where to save the figure

Each backend is benchmarked in its own subprocess. This isolates them (one
backend crashing can't take down the others) and, importantly, keeps
``torchvulkan`` imported *only* in the vulkan worker: registering its
PrivateUse1 backend otherwise breaks MPS ``backward()`` in the same process.

Data is synthetic (random tensors shaped like MNIST batches), so no dataset
download is required.
"""

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import warnings

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from examples.mnist import SimpleNN

ALL_BACKENDS = ["cpu", "cuda", "mps", "vulkan"]
RESULT_PREFIX = "RESULT_JSON "
DTYPE = torch.float16


def param_count(hidden: int) -> int:
    return 784 * hidden + hidden + hidden * 10 + 10


def device_available(backend: str) -> bool:
    if backend == "cpu":
        return True
    if backend == "cuda":
        return torch.cuda.is_available()
    if backend == "mps":
        mps = getattr(torch.backends, "mps", None)
        return mps is not None and mps.is_available()
    if backend == "vulkan":
        try:
            import torchvulkan

            return torchvulkan.is_available()
        except Exception:  # noqa: BLE001 - torchvulkan is optional
            return False
    return False


def sync_fn(backend: str):
    if backend == "cuda":
        return torch.cuda.synchronize
    if backend == "mps":
        return torch.mps.synchronize
    if backend == "vulkan":
        import torchvulkan

        return torchvulkan.synchronize
    return lambda: None


def time_pass(model, criterion, data, targets, sync, iters, warmup):
    """Median combined forward + backward time (ms) over `iters` runs."""
    times = []
    for i in range(warmup + iters):
        model.zero_grad(set_to_none=True)

        sync()
        t0 = time.perf_counter()
        loss = criterion(model(data), targets)
        loss.backward()
        sync()
        t1 = time.perf_counter()

        if i >= warmup:  # discard warmup iterations
            times.append((t1 - t0) * 1e3)

    return statistics.median(times)


def empty_cache(backend):
    if backend == "cuda":
        torch.cuda.empty_cache()
    elif backend == "mps":
        torch.mps.empty_cache()
    elif backend == "vulkan":
        import torchvulkan

        torchvulkan.empty_cache()


def worker(backend, sizes, batch_size, iters, warmup):
    """Benchmark one backend; emit rows as JSON on stdout, progress on stderr."""
    warnings.filterwarnings("ignore")  # hide one-time CPU-fallback notices
    result = {"backend": backend, "available": False, "rows": []}

    if not device_available(backend):
        print(f"[{backend}] not available on this machine", file=sys.stderr)
        print(RESULT_PREFIX + json.dumps(result))
        return

    result["available"] = True
    device = torch.device(backend)
    sync = sync_fn(backend)
    criterion = nn.CrossEntropyLoss()

    for hidden in sizes:
        params = param_count(hidden)
        model = data = targets = None
        try:
            torch.manual_seed(0)
            model = SimpleNN(hidden).to(device).to(DTYPE)
            data = torch.randn(batch_size, 1, 28, 28, device=device, dtype=DTYPE)
            targets = torch.randint(0, 10, (batch_size,), device=device)

            total = time_pass(model, criterion, data, targets, sync, iters, warmup)
            result["rows"].append([params, total])
            print(
                f"[{backend}] hidden={hidden:<7} ~{params:>12,} params | "
                f"{total:9.3f} ms",
                file=sys.stderr,
            )
        except Exception as e:
            result["rows"].append([params, None])
            print(f"[{backend}] hidden={hidden:<7} failed: "
                  f"{str(e)[:80]}", file=sys.stderr)
        finally:  # free the tensors before the next size
            del model, data, targets
            gc.collect()
            empty_cache(backend)

    print(RESULT_PREFIX + json.dumps(result))


def spawn_worker(backend, sizes, batch_size, iters, warmup):
    """Run one backend in a fresh process; return its rows (or None)."""
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--worker", backend,
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--sizes", *[str(s) for s in sizes],
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            payload = json.loads(line[len(RESULT_PREFIX):])
            if payload["available"] and any(r[1] is not None for r in payload["rows"]):
                return payload["rows"]
            return None
    print(f"[{backend}] worker produced no result (exit {proc.returncode})",
          file=sys.stderr)
    return None


def candidate_backends():
    """Backends worth probing on this machine (vulkan decided by its worker)."""
    cands = ["cpu"]
    if torch.cuda.is_available():
        cands.append("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        cands.append("mps")
    cands.append("vulkan")
    return cands


PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"


def plot(results, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(results)}

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for name, rows in results.items():
        rows = [r for r in rows if r[1] is not None]
        if not rows:
            continue
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        ax.plot(xs, ys, marker="o", ms=6, lw=2, color=colors[name], label=name)
        ax.annotate(
            name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=11, color=colors[name], fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("model parameters", fontsize=11, color=MUTED)
    ax.set_ylabel("time per fwd + bwd pass (ms)", fontsize=11, color=MUTED)
    ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.7)
    ax.tick_params(colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.legend(frameon=False, fontsize=11, labelcolor=INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    print(f"\nSaved plot -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sizes", type=int, nargs="+",
        default=[64, 1024, 16384, 65536, 131072, 262144, 393216],
        help="hidden widths to sweep (params ~= 795 x width; the default top "
             "reaches ~312M params)",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--iters", type=int, default=10, help="timed iters per point")
    p.add_argument("--warmup", type=int, default=3, help="untimed warmup iters")
    p.add_argument(
        "--out", default=os.path.join(os.path.dirname(__file__), "media/benchmark.png")
    )
    p.add_argument(
        "--worker", choices=ALL_BACKENDS,
        help="internal: benchmark a single backend and print JSON",
    )
    args = p.parse_args()

    if args.worker:
        worker(args.worker, args.sizes, args.batch_size, args.iters, args.warmup)
        return

    print("Probing backends:", ", ".join(candidate_backends()), flush=True)
    print(f"batch_size={args.batch_size}, iters={args.iters}, "
          f"warmup={args.warmup}\n", flush=True)

    results = {}
    for backend in candidate_backends():
        rows = spawn_worker(
            backend, args.sizes, args.batch_size, args.iters, args.warmup
        )
        if rows is not None:
            results[backend] = rows

    if not results:
        print("No backend produced results.")
        return

    print("\nBenchmarked backends:", ", ".join(results))
    plot(results, args.out)


if __name__ == "__main__":
    main()
