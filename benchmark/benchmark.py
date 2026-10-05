"""Benchmark a model's forward + backward pass across every available backend.

Detects the compute backends on this machine (CPU, CUDA, MPS, and the
torchvulkan `vulkan` device), then times a combined forward + backward pass
(``float16`` on accelerators, ``float32`` on CPU) at a range of model sizes, and plots the
per-backend timings against parameter count so you can see how each
implementation scales as the model grows.

    python benchmark/benchmark.py mnist                 # sensible defaults
    python benchmark/benchmark.py llm --iters 50        # more samples per point
    python benchmark/benchmark.py mnist --out bench.png # where to save the figure

The model is selected by subcommand (``mnist`` or ``llm``); everything else
-- backend detection, subprocess isolation, timing, plotting -- is shared.
Adding a new model to benchmark means writing one ``Task`` subclass, not a
second copy of this whole file.

Each backend is benchmarked in its own subprocess. This isolates them (one
backend crashing can't take down the others) and, importantly, keeps
``torchvulkan`` imported *only* in the vulkan worker: registering its
PrivateUse1 backend otherwise breaks MPS ``backward()`` in the same process.

Data is synthetic (random tensors), so no dataset or tokenizer is needed.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.mnist import SimpleNN
from examples.llm import TinyGPT

ALL_BACKENDS = ["cpu", "cuda", "mps", "vulkan"]
RESULT_PREFIX = "RESULT_JSON "
DTYPE = torch.float16
# CPU half-precision kernels are far slower than float32 ones, so fp16 would misrepresent the CPU baseline
BACKEND_DTYPES = {"cpu": torch.float32}


class Task:
    """One model to benchmark: how to size it, build it, and score it.

    The harness below (backend detection, subprocess isolation, timing,
    plotting) is the same for every task; only these five methods vary.
    """

    name = "override-me"

    def add_args(self, parser):
        """Add this task's CLI arguments (at minimum: --sizes, --batch-size)."""
        raise NotImplementedError

    def cli_args(self, args):
        """This task's arguments, re-serialized to forward to a worker subprocess."""
        raise NotImplementedError

    def param_count(self, size, args) -> int:
        raise NotImplementedError

    def build(self, size, args, device, dtype):
        """Return (model, data, targets) for one benchmark point."""
        raise NotImplementedError

    def loss(self, model, data, targets, criterion):
        """Run the forward pass and return the loss tensor."""
        raise NotImplementedError


class MnistTask(Task):
    name = "mnist"

    def add_args(self, parser):
        parser.add_argument(
            "--sizes", type=int, nargs="+",
            default=[64, 1024, 16384, 65536, 131072, 262144, 393216],
            help="hidden widths to sweep (params ~= 795 x width; the default "
                 "top reaches ~312M params)",
        )
        parser.add_argument("--batch-size", type=int, default=128)

    def cli_args(self, args):
        return ["--batch-size", str(args.batch_size)]

    def param_count(self, size, args):
        return 784 * size + size + size * 10 + 10

    def build(self, size, args, device, dtype):
        model = SimpleNN(size).to(device).to(dtype)
        data = torch.randn(args.batch_size, 1, 28, 28, device=device, dtype=dtype)
        targets = torch.randint(0, 10, (args.batch_size,), device=device)
        return model, data, targets

    def loss(self, model, data, targets, criterion):
        return criterion(model(data), targets)


class LLMTask(Task):
    name = "llm"
    HEAD_DIM = 64

    def add_args(self, parser):
        parser.add_argument(
            "--sizes", type=int, nargs="+",
            default=[64, 128, 256, 384, 512, 768],
            help="n_embd (model width) values to sweep, each a multiple of "
                 "the 64-wide head size; the default top reaches ~30M params",
        )
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--seq-len", type=int, default=64, help="context length")
        parser.add_argument("--n-layer", type=int, default=4, help="transformer blocks, held fixed while n_embd is swept")
        parser.add_argument("--vocab-size", type=int, default=1000)

    def cli_args(self, args):
        return [
            "--batch-size", str(args.batch_size),
            "--seq-len", str(args.seq_len),
            "--n-layer", str(args.n_layer),
            "--vocab-size", str(args.vocab_size),
        ]

    def param_count(self, size, args):
        # attention isn't shaped like SDPA needs (no new shader): torch.bmm, not the general
        # broadcasting torch.matmul, whose backward isn't wired up for every backend yet.
        with torch.device("meta"):
            model = TinyGPT(
                vocab_size=args.vocab_size, block_size=args.seq_len,
                n_layer=args.n_layer, n_embd=size, head_dim=self.HEAD_DIM,
            )
        return sum(p.numel() for p in model.parameters())

    def build(self, size, args, device, dtype):
        model = TinyGPT(
            vocab_size=args.vocab_size, block_size=args.seq_len,
            n_layer=args.n_layer, n_embd=size, head_dim=self.HEAD_DIM,
        ).to(device).to(dtype)
        data = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
        targets = torch.randint(0, args.vocab_size, (args.batch_size * args.seq_len,), device=device)
        return model, data, targets

    def loss(self, model, data, targets, criterion):
        logits = model(data)
        return criterion(logits.view(-1, logits.size(-1)), targets)


TASKS = {"mnist": MnistTask(), "llm": LLMTask()}


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


def time_pass(model, loss_fn, sync, iters, warmup):
    """Median combined forward + backward time (ms) over `iters` runs."""
    times = []
    for i in range(warmup + iters):
        model.zero_grad(set_to_none=True)

        sync()
        t0 = time.perf_counter()
        loss_fn().backward()
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


def worker(task, backend, args):
    """Benchmark one backend on `task`; emit rows as JSON on stdout, progress on stderr."""
    warnings.filterwarnings("ignore")  # hide one-time CPU-fallback notices
    result = {"backend": backend, "available": False, "rows": []}

    if not device_available(backend):
        print(f"[{backend}] not available on this machine", file=sys.stderr)
        print(RESULT_PREFIX + json.dumps(result))
        return

    result["available"] = True
    device = torch.device(backend)
    dtype = BACKEND_DTYPES.get(backend, DTYPE)
    sync = sync_fn(backend)
    criterion = nn.CrossEntropyLoss()

    for size in args.sizes:
        params = task.param_count(size, args)
        model = data = targets = None
        try:
            torch.manual_seed(0)
            model, data, targets = task.build(size, args, device, dtype)

            total = time_pass(model, lambda: task.loss(model, data, targets, criterion), sync, args.iters, args.warmup)
            result["rows"].append([params, total])
            print(
                f"[{backend}] size={size:<7} ~{params:>12,} params | "
                f"{total:9.3f} ms",
                file=sys.stderr,
            )
        except Exception as e:
            result["rows"].append([params, None])
            print(f"[{backend}] size={size:<7} failed: "
                  f"{str(e)[:80]}", file=sys.stderr)
        finally:  # free the tensors before the next size
            del model, data, targets
            gc.collect()
            empty_cache(backend)

    print(RESULT_PREFIX + json.dumps(result))


def spawn_worker(task, backend, args):
    """Run one backend in a fresh process; return its rows (or None)."""
    cmd = [
        sys.executable, os.path.abspath(__file__), args.model,
        "--worker", backend,
        "--iters", str(args.iters),
        "--warmup", str(args.warmup),
        *task.cli_args(args),
        "--sizes", *[str(s) for s in args.sizes],
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
        label = f"{name} ({str(BACKEND_DTYPES.get(name, DTYPE)).removeprefix('torch.')})"
        ax.plot(xs, ys, marker="o", ms=6, lw=2, color=colors[name], label=label)
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


def build_parser():
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--iters", type=int, default=10, help="timed iters per point")
    shared.add_argument("--warmup", type=int, default=3, help="untimed warmup iters")
    shared.add_argument("--out", default=None, help="plot path (default: media/<model>_benchmark.png)")
    shared.add_argument(
        "--worker", choices=ALL_BACKENDS,
        help="internal: benchmark a single backend and print JSON",
    )

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="model", required=True, help="which model to benchmark")
    for name, task in TASKS.items():
        task_parser = sub.add_parser(name, parents=[shared], help=f"benchmark {task.name}")
        task.add_args(task_parser)
    return p


def main():
    args = build_parser().parse_args()
    task = TASKS[args.model]

    if args.worker:
        worker(task, args.worker, args)
        return

    print("Probing backends:", ", ".join(candidate_backends()), flush=True)
    print(f"model={args.model}, iters={args.iters}, warmup={args.warmup}\n", flush=True)

    results = {}
    for backend in candidate_backends():
        rows = spawn_worker(task, backend, args)
        if rows is not None:
            results[backend] = rows

    if not results:
        print("No backend produced results.")
        return

    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media", f"{args.model}_benchmark.png")
    print("\nBenchmarked backends:", ", ".join(results))
    plot(results, out)


if __name__ == "__main__":
    main()
