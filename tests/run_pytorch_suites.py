"""
Runs PyTorch's own device-generic test files on the vulkan device.

PyTorch's tests are written once and instantiated per device type by
instantiate_device_type_tests; with torchvulkan imported first, each file gets
a TestFooPRIVATEUSE1 class whose tests run on "vulkan". The torch wheel doesn't
ship the test files, so `fetch` checks out PyTorch's test/ at the installed
torch's commit.

    python tests/run_pytorch_suites.py fetch
    python tests/run_pytorch_suites.py list [FILE ...]
    python tests/run_pytorch_suites.py run [FILE ...] [--no-strict] [--timeout S] [-- PYTEST_ARGS]
    python tests/run_pytorch_suites.py summary

PyTorch's tests need pytest, expecttest, hypothesis, pyyaml and numpy.

With no FILE, every device-generic file except EXCLUDED_DIRS/EXCLUDED_FILES.
Each file runs in its own pytest process, under TORCHVULKAN_STRICT=1 unless
--no-strict, appending each test's outcome to <out>/<file>.jsonl. A test that
crashes the process or runs past --timeout is recorded as such and pytest
starts again with the tests not yet run (so does a lost device, which would
fail every later test); rerunning `run` resumes the same way.
Failures are then rerun on the CPU: those that fail there too are counted as
"fails on CPU too", not as vulkan bugs. CPU fallbacks in a dtype the device
can't compute in (probed into <out>/device.json) are "unsupported dtype", and
failures in EXPECTED are "expected". `summary` writes <out>/summary.md and
<out>/failures.txt.
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXCLUDED_DIRS = {
    "distributed": "needs process groups",
    "dynamo": "torch.compile",
    "inductor": "torch.compile: Inductor has no vulkan code generator",
    "export": "torch.compile",
    "higher_order_ops": "torch.compile",
    "xpu": "another device",
    "onnx": "ONNX export",
    "lazy": "lazy tensors",
    "cpp_extensions": "builds C++ extensions",
}

EXCLUDED_FILES = {
    "test_cuda.py": "CUDA only",
    "test_matmul_cuda.py": "CUDA only",
    "test_scaled_matmul_cuda.py": "CUDA only",
    "test_jiterator.py": "CUDA only",
    "test_mps.py": "MPS only",
    "test_xpu.py": "XPU only",
    "test_dynamic_shapes.py": "torch.compile",
    "test_precompile.py": "torch.compile",
    "test_compile_benchmark_util.py": "torch.compile",
    "nn/attention/test_fa3.py": "FlashAttention 3 on CUDA",
    "nn/attention/test_fa4.py": "FlashAttention 4 on CUDA",
    "profiler/test_trace_validator.py": "the profiler has no activity for PrivateUse1 devices",
}

# a class is named after PrivateUse1TestBase.device_type, which setUpClass renames from "privateuse1" to "vulkan"
VULKAN_CLASSES = "PRIVATEUSE1,VULKAN"
# a lost device (VK_ERROR_DEVICE_LOST) fails every later test in its process: the worker exits and pytest starts again
DEVICE_LOST = "VkResult -4"
RESTART = 75
FALLBACK = "torchvulkan [NOT IMPLEMENTED]"

# dtypes a device may lack in hardware (Apple GPUs: float64, bfloat16), and how messages and test names spell them;
# their CPU fallbacks are counted as "unsupported dtype", not as missing kernels
DEVICE_DTYPES = {
    "float64": ("Double", "ComplexDouble", "float64", "complex128"),
    "bfloat16": ("BFloat16", "bfloat16"),
    "float16": ("Half", "float16"),
    "int64": ("Long", "int64", "uint64"),
    "int16": ("Short", "int16", "uint16"),
    "int8": ("Char", "int8"),
}

# failures that no PrivateUse1 backend can avoid: (test id pattern, message pattern, reason)
EXPECTED = [
    (r"test_convolution\.py::.*::test_conv_backend_", r"_ConvBackend\.Overrideable",
     "torch._C._select_conv_backend returns Overrideable for every out-of-tree device"),
    (r"test_embedding\.py::.*::test_embedding_bag_out_of_bounds_idx_", r"RuntimeError not raised",
     "no synchronous index bounds check on the GPU (the test skips CUDA and XPU for the same reason)"),
    (r".*", r"Unknown device type vulkan for Dlpack",
     "torch's Tensor.__dlpack_device__ maps PrivateUse1 only under its default name, not a renamed backend"),
    (r".*", r"InductorError|no scheduling registered for device vulkan|Device vulkan not supported",
     "torch.compile's default Inductor backend has no vulkan code generator"),
    (r"test_distributions\.py::.*::test_(multivariate_normal|wishart)_moments_", r"Tensor-likes are not close",
     "a fixed-seed statistical check tuned to the CPU's random stream; the vulkan samples are unbiased (mean z-scores "
     "~N(0, 1) over 20 seeds) but come from Philox"),
    (r"test_binary_ufuncs\.py::.*::test_cpu_tensor_pow_device_scalar_tensor_", r"Expected all tensors to be on the same device",
     "PyTorch's own pow_ kernel rejects a CPU self with an accelerator exponent; the test expects that only from CUDA and XPU"),
    (r"test_autograd\.py::.*::test_checkpointing_non_reentrant_autocast_", r"torch\.bfloat16 != torch\.float32",
     "autocast disables itself for bfloat16 on a device without it (the test skips such CUDA and XPU devices)"),
    (r"test_autograd\.py::.*::test_lookup_does_not_deadlock_with_python_dispatch_", r"",
     "the test's subprocess script hard-codes device=\"cuda\""),
    (r"test_ops\.py::.*::test_fake_crossref_backward_.*normal_number_mean_", r"flat_diff_results",
     "normal with only Python-number inputs has nothing to differentiate (PyTorch xfails it for MPS)"),
    (r"test_indexing\.py::.*::test_trivial_fancy_out_of_bounds_", r"IndexError not raised",
     "no synchronous index bounds check on the GPU (the test skips CUDA and XPU for the same reason)"),
    (r"test_indexing\.py::.*::test_index_add_smem_stage_alignment_regression_", r"Unsupported device type",
     "the test reads CUDA/XPU-only device properties and raises for any other device"),
    (r"test_indexing\.py::.*::test_index_put_deterministic_with_optional_tensors_", r'"must match" does not match "shape mismatch',
     "the test expects CUDA's deterministic-path wording; the error is raised, with the CPU's message"),
    (r"test_linalg\.py::.*::test_svd_vulkan_", r"KeyError: 'vulkan",
     "the test looks up SVD drivers in a dict with only cpu, cuda and xpu keys"),
    (r"test_ops\.py::.*::test_compare_cpu_nn_functional_(alpha|feature_alpha)_dropout_", r"Tensor-likes are not close",
     "a random op compared against the CPU's random stream (training=True samples)"),
    (r"test_nn\.py::.*::test_upsamplingBiMode2d_nonsupported_dtypes_.*uint8", r"RuntimeError not raised",
     "uint8 bilinear/bicubic (and antialiased) resampling is implemented here, as on the CPU; the test expects only the CPU to have it"),
    (r"test_nn\.py::.*::test_transformerencoderlayer_vulkan", r"False is not true",
     "torch takes the transformer fast path for PrivateUse1 too (NaN for fully masked rows), but the test treats only CUDA and CPU as fast-path devices"),
    (r"test_multiprocessing\.py::.*::test_wrong_device_fork_", r"Cannot re-initialize VULKAN",
     "the test's subprocess script never imports torchvulkan (torch has no autoload entry point for it), so 'vulkan' is unknown there"),
    (r"test_ops\.py::.*::test_python_ref_errors__refs_where_", r"is not on the expected device",
     "torch's _refs.where raises its own device message on FakeTensors; the OpInfo xfails it for CUDA, XPU and MPS"),
    (r"test_ops\.py::.*::test_python_ref_executor__refs_nn_functional_alpha_dropout_executor_aten_.*float16",
     r"Dtypes torch\.float32 and torch\.float16", "torch's alpha_dropout ref returns float32 for a half input (a bool mask times a Python "
     "float promotes); the OpInfo skips this test on CUDA and XPU"),
    (r"test_ops\.py::.*::test_python_ref_executor__refs_geometric_executor_aten_.*int", r"check_uniform_bounds",
     "torch's geometric ref draws uniform_ in the integer dtype, which torch rejects on every device; the OpInfo skips this test on "
     "CUDA and XPU"),
    (r"test_ops\.py::.*::test_python_ref_executor__refs_nn_functional_nll_loss_executor_aten_",
     r"get value out of a tracing tensor", "torch's nll_loss ref indexes with a 0-dim target, an .item() that make_fx can't trace "
     "for an accelerator tensor; the OpInfo xfails this test on CUDA"),
    (r"test_ops\.py::.*::test_fake_autocast_linalg_pinv_", r"Dtypes torch\.float32 and torch\.float16",
     "under autocast linalg.pinv's inner matmuls drop to half while its meta function stays float32; the test skips pinv on "
     "every in-tree device"),
    (r"test_ops\.py::.*::test_(fake|fake_autocast|fake_crossref_backward_(no_)?amp)_nn_functional_grid_sample_",
     r"bicubic interpolation with 5D input is not supported on vulkan",
     "torch's composite grid_sampler allows 5-D bicubic only on its own devices (the vulkan kernel handles it), and the "
     "composite can't be overridden without breaking autograd"),
    (r"test_nn\.py::.*::test_grid_sample_3d_bicubic_last_voxel_of_a_wide_axis_size_9007199254740993_", r"OutOfMemoryError",
     "the kernel reads a dense input, and the test's 2**53 + 1 element stride-0 view would need tens of petabytes once materialized"),
    (r"test_foreach\.py::.*::test_big_num_tensors_.*_use_accelerator_graph_True_", r"dummy base class CUDAGraph",
     "the graph variant builds torch.cuda.CUDAGraph for every accelerator but XPU"),
    (r"test_foreach\.py::.*::test_foreach_reduce_large_input__foreach_norm_.*_(bool|u?int\d+)$",
     r"linalg\.vector_norm: Expected a floating point",
     "the reference linalg.vector_norm rejects integer tensors on every device; the OpInfo xfails this test for CUDA and XPU"),
    (r"test_foreach\.py::.*::test_unary_op_tensors_on_different_devices__foreach_neg_.*_bool$", r"does not match",
     "the test matches the error text as a regex, and the parentheses in its `logical_not()` don't match themselves; the "
     "OpInfo xfails this test for CUDA and XPU"),
    (r"test_masked\.py::.*::test_mask_layout_sparse_coo_masked_prod_.*_float16$", r"Scalars are not close",
     "the dense reference reduces one dim at a time, rounding to half between them, and lands 2 ulps (1.1e-3) from the "
     "product; the sparse result is the correctly rounded one"),
    (r"test_modules\.py::.*::test_cpu_gpu_parity_nn_CrossEntropyLoss_.*_float16$", r"not close",
     "float16 gradients round differently from the CPU's at the test's 1e-5 tolerance (both ~5e-4 from float64); the "
     "ModuleInfo xfails this test for CUDA and XPU"),
    (r"test_modules\.py::.*::test_cpu_gpu_parity_nn_ConvTranspose[123]d_.*_complex32$", r"not close",
     "the CPU has no complex32 convolution to match; the ModuleInfo xfails this test for CUDA and skips it for XPU"),
    (r"test_nn\.py::.*::test_linear_cross_entropy_loss_prob_target_.*", r"not less than or equal to",
     "PyTorch calibrates these drift caps per backend from observed values (float32: CPU 16384/384 ULP, MPS 256, CUDA "
     "512/128); vulkan gets CUDA's and measures 673/275 ULP in float32, inside the CPU's caps"),
    (r"test_unary_ufuncs\.py::.*::test_reference_numerics_(small|large|extremal)_(_refs_)?atan_.*_complex64$", r"not close",
     "the reference is macOS's libm catanf, which differs at the branch points (atan(-i) = pi/4 - inf i); the OpInfo "
     "skips these on macOS for every device, and vulkan matches torch's CPU there"),
    (r"test_unary_ufuncs\.py::.*::test_reference_numerics_extremal_(_refs_)?atanh_.*_complex64$", r"not close",
     "NumPy gives NaN + i pi/2 for atanh(NaN + i inf) where C99 allows +-0; the OpInfo skips this test on the CPU, and "
     "vulkan matches the CPU's +-0"),
    (r"test_spectral_ops\.py::.*::test_(fft_round_trip|fft_type_promotion|fftn_noop_transform|fftn_round_trip|hfftn|ihfftn)_.*_float16$",
     r"Unsupported dtype Half",
     "torch's promote_type_fft accepts real float16 only on CUDA, XPU and meta devices, before any backend kernel runs"),
    (r"test_sort_and_select\.py::.*::test_unique_dim_", r"not close",
     "the NaN case: torch's unique_dim sorts rows with a comparator that calls NaN equal to everything (no strict weak "
     "order), so the expected rows are std::sort's and thrust's visiting order, not a defined result"),
    (r"test_testing\.py::.*::test_get_supported_dtypes_", r"Expected device_type in \['cpu', 'cuda'\]",
     "torch.testing's opinfo.utils.get_supported_dtypes accepts only cpu and cuda"),
    (r"test_testing\.py::.*::test_setup_and_teardown_run_for_device_specific_tests_", r"not found in",
     "the test's subprocess imports only torch, and the vulkan device registers when torchvulkan is imported (no "
     "torch.backends autoload entry point), so it instantiates no tests"),
    (r"test_view_ops\.py::.*::test_(flatten|ravel)_", r"_empty_affine_quantized.*Quantizedvulkan",
     "quantized tensors: torch has no quantized kernels for an out-of-tree backend, and torchvulkan doesn't implement them"),
    (r"test_ops\.py::.*::test_python_ref_executor__refs_linspace_executor_aten_.*_(u?int(8|16|32|64))$",
     r"cannot be safely cast to type",
     "torch's _refs.linspace can't be traced for integer dtypes; the OpInfo xfails this test for CUDA and XPU"),
    (r"test_ops_jit\.py::.*::test_variant_consistency_jit_nn_functional_cross_entropy_.*_float32$", r"Scalars are not close",
     "the scripted variant embeds the weight kwarg as its printed literal (4 decimals); this sample's weights nearly cancel "
     "(sum 0.576), so the rounding alone moves the loss 3e-5. The eager result equals the CPU's bit for bit"),
    (r"test_torch\.py::.*::test_tensor_storage_type_", r"Type comparison failed: <class 'torch\.vulkan\.",
     "the test expects the CPU's storage class off CUDA, but torch's _get_legacy_storage_class looks it up on the "
     "PrivateUse1 module (torch.vulkan.FloatStorage, ...)"),
    (r"test_sparse\.py::.*::test_reductions_sum_SparseCOO_.*_float16$", r"not close",
     "torch's sum_sparse_coo adds duplicates in float16 steps and the dense sum in a float accumulator: a few float16 ulps "
     "apart on the CPU too (0.016 there with seed 0, past this tolerance), so passing depends on the samples"),
    (r"test_sparse\.py::.*::test_coalesce_accepts_large_tensor_", r"VkResult -4|OutOfMemoryError",
     "272.5M nonzeros, which the test sizes at 30 GB for CUDA (it skips MPS)"),
    (r"test_sparse_csr\.py::.*::test_sparse_mm_reduce", r"_sparse_mm_reduce_impl",
     "torch implements _sparse_mm_reduce_impl for SparseCsrCPU alone; the test skips CUDA for that reason"),
    (r"test_torch\.py::.*::test_broadcast_fn_map2?_", r"is only implemented on CPU tensors",
     "the test skips map_/map2_ only for CUDA tensors; torch implements them for CPU tensors alone"),
    (r"test_torch\.py::.*::test_grad_scaler_deprecated_warning_", r"torch\.cpu\.amp\.GradScaler",
     "the test builds torch.cpu.amp.GradScaler off CUDA but expects the warning to name the device type"),
    (r"test_torch\.py::.*::test_grad_scaling_state_dict_", r"False is not true",
     "the test expects a lazily made scale to be a torch.FloatTensor (a CPU tensor type) on every device but CUDA"),
    (r"test_torch\.py::.*::test_scatter_reduce_multiply_unsupported_dtypes_.*_complex", r"RuntimeError not raised",
     "the test asserts every accelerator lacks complex scatter multiply (CUDA has no kernel); vulkan multiplies in "
     "source order as the CPU does, which test_scatter_gather_ops checks"),
    (r"test_optim\.py::.*::test_peak_memory_foreach_", r"has no attribute 'current_device'",
     "the test reads torch.cuda's memory stats, which on another accelerator reach torch.mps"),
    (r"test_optim\.py::.*::test_foreach_large_tensor_", r"OutOfMemoryError|Out of memory|Can't fill a Half tensor of this shape",
     "2**32-element float16 params and grads, which the test sizes at 72 GB for CUDA (it skips MPS)"),
    (r"test_optim\.py::.*::test_forloop_goes_right_direction_contiguous_False_.*_LBFGS_", r"nan not greater than",
     "maximize=True with a strong-Wolfe line search overflows float32 within two steps on this unbounded objective; "
     "whether it ends at inf or nan follows float32 summation order (the same data takes the same path on both "
     "devices until the low bits differ)"),
    (r"test_opaque_obj_v2\.py::.*::test_make_fx_with_generator_", r"_scaled_dot_product_efficient_attention",
     "the test calls aten::_scaled_dot_product_efficient_attention directly, CUDA's and XPU's kernel (the CPU has none)"),
    (r"test_dims\.py::.*::test_attn_device_", r"OutOfMemoryError",
     "first-class dims multiply q * k before summing at BERT size: a 256x12x128x128x64 float32 product, 12 GiB, on any "
     "device"),
    (r"test_multiprocessing\.py::.*::test_(simple|variable|parameter)_sharing_", r"AssertionError: (tensor\(False.*)?(False )?is not true",
     "Vulkan memory isn't shared between processes: a tensor crosses by value, so the child's writes stay in the child"),
    (r".*", r"^(Failed: )?Unexpected success",
     "an OpInfo expected failure (marked for the CPU and CUDA kernels) passes here"),
    (r"test_content_store\.py::.*::test_load_tensor_", r"torch\.float32 != torch\.float64",
     "loading as float64 copies Float to Double, which a device without float64 can't do (the test skips MPS for this); "
     "dynamo's clone_input swallows the strict-mode error and returns float32"),
]



def cache_dir():
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "torchvulkan"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "torchvulkan"


def default_tests_dir():
    import torch
    return cache_dir() / "pytorch" / torch.version.git_version / "test"


def fetch(tests_dir):
    import torch
    commit = torch.version.git_version
    repo = tests_dir.parent
    if tests_dir.is_dir():
        print(f"PyTorch tests for torch {torch.__version__} ({commit[:12]}) already in {tests_dir}")
        return
    repo.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True)
    git("init", "-q")
    git("remote", "add", "origin", "https://github.com/pytorch/pytorch.git")
    git("config", "core.sparseCheckout", "true")
    (repo / ".git" / "info" / "sparse-checkout").write_text("test/\n")
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
    git("checkout", "-q", "FETCH_HEAD")
    print(f"PyTorch tests for torch {torch.__version__} ({commit[:12]}) in {tests_dir}")


def discover(tests_dir):
    files = []
    for path in sorted(tests_dir.rglob("test_*.py")):
        rel = path.relative_to(tests_dir).as_posix()
        if rel.split("/")[0] in EXCLUDED_DIRS or rel in EXCLUDED_FILES:
            continue
        if "instantiate_device_type_tests" in path.read_text(encoding="utf-8", errors="ignore"):
            files.append(rel)
    return files


def cpu_twin(nodeid):
    """TestFooPRIVATEUSE1::test_x_vulkan_float32 -> TestFooCPU::test_x_cpu_float32"""
    file, cls, method = nodeid.split("::", 2)
    cls = re.sub(r"(PRIVATEUSE1|VULKAN)$", "CPU", cls)
    return f"{file}::{cls}::{re.sub(r'_vulkan(?=_|$|\[)', '_cpu', method)}"


# ---------------------------------------------------------------- worker (one pytest process)

class Recorder:
    def __init__(self, jsonl, class_suffix, only, skip):
        self.out = open(jsonl, "a", encoding="utf-8")
        self.class_suffixes = tuple(class_suffix.split(","))
        self.only = only
        self.skip = skip
        self.started = time.time()

    def write(self, **entry):
        self.out.write(json.dumps(entry) + "\n")
        self.out.flush()

    def keep(self, nodeid):
        parts = nodeid.split("::")
        if len(parts) < 3 or not parts[1].endswith(self.class_suffixes):
            return False
        return (self.only is None or nodeid in self.only) and nodeid not in self.skip

    def pytest_collection_modifyitems(self, config, items):
        kept = [item for item in items if self.keep(item.nodeid)]
        dropped = [item for item in items if not self.keep(item.nodeid)]
        if dropped:
            config.hook.pytest_deselected(items=dropped)
        items[:] = kept

    def pytest_runtest_logstart(self, nodeid, location):
        self.started = time.time()
        self.write(start=nodeid, time=self.started)

    def pytest_runtest_teardown(self, item):
        # errors a kernel reports asynchronously belong to this test, not the next one to synchronize
        if "CPU" not in self.class_suffixes:
            import torchvulkan
            try:
                torchvulkan.synchronize()
            except Exception as e:
                self.write(nodeid=item.nodeid, outcome="failed", message=f"at teardown synchronize: {e}"[:300])
                if DEVICE_LOST in str(e):
                    os._exit(RESTART)

    def pytest_runtest_logreport(self, report):
        if report.when != "call" and report.passed:
            return
        outcome = report.outcome
        if hasattr(report, "wasxfail"):
            outcome = "xfailed" if report.skipped else "xpassed"
        message = ""
        if report.failed:
            crash = getattr(report.longrepr, "reprcrash", None)
            message = crash.message if crash is not None else str(report.longrepr)
        elif report.skipped and isinstance(report.longrepr, tuple):
            message = report.longrepr[2]
        lines = message.strip().splitlines()
        first = lines[0] if lines else ""
        if report.failed and FALLBACK not in first:
            # a fallback can be wrapped (torch.testing's "Comparing ..." chains the original error)
            wrapped = [line for line in str(report.longrepr).splitlines() if FALLBACK in line]
            if wrapped:
                first = wrapped[0].lstrip("E ").strip()
        self.write(nodeid=report.nodeid, outcome=outcome, message=first[:300],
                   elapsed=round(time.time() - self.started, 3))
        if DEVICE_LOST in message:
            os._exit(RESTART)


def worker(argv):
    sep = argv.index("--")
    jsonl, class_suffix, only_file, skip_file = argv[:sep]
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
    import torchvulkan  # noqa: F401  registers "vulkan" before common_device_type lists the device types
    import pytest
    import torch
    if torch._C._get_privateuse1_backend_name() != "vulkan":
        sys.exit(f"torchvulkan didn't register the vulkan device (imported {torchvulkan.__file__})")
    read_ids = lambda f: set(Path(f).read_text().split("\n")) - {""} if f else None
    recorder = Recorder(jsonl, class_suffix, read_ids(only_file), read_ids(skip_file) or set())
    sys.exit(pytest.main(argv[sep + 1:], plugins=[recorder]))


# ---------------------------------------------------------------- runner

def read_jsonl(path):
    entries = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut short by a crash
    return entries


def outcomes(path):
    """{nodeid: last entry with an outcome}, and the start entry of the test still running, if any"""
    results, running = {}, None
    for e in read_jsonl(path):
        if "start" in e:
            running = e
        elif "nodeid" in e:
            results[e["nodeid"]] = e
            if running is not None and e["nodeid"] == running["start"]:
                running = None
    return results, running


def run_pytest(test_file, tests_dir, jsonl, log, env, class_suffix, timeout, pytest_args, only=None):
    """Runs test_file until every selected test has an outcome, recording crashes and timeouts."""
    scratch = jsonl.with_suffix(".ids")
    while True:
        done, _ = outcomes(jsonl)
        scratch.write_text("\n".join(done))
        only_file = ""
        if only is not None:
            only_file = str(jsonl.with_suffix(".only"))
            Path(only_file).write_text("\n".join(only))
        cmd = [sys.executable, str(Path(__file__).resolve()), "_worker", str(jsonl), class_suffix, only_file,
               str(scratch), "--", test_file, "-q", "-p", "no:cacheprovider", *pytest_args]
        with open(log, "a", encoding="utf-8") as out:
            offset = jsonl.stat().st_size if jsonl.exists() else 0
            proc = subprocess.Popen(cmd, cwd=tests_dir, env=env, stdout=out, stderr=subprocess.STDOUT)
            timed_out, running = False, None
            while proc.poll() is None:
                time.sleep(2)
                if not jsonl.exists():
                    continue  # the worker is still starting
                # only the lines written since the last look, so a long file costs nothing per poll
                with open(jsonl, "rb") as f:
                    f.seek(offset)
                    chunk = f.read()
                chunk = chunk[:chunk.rfind(b"\n") + 1]
                offset += len(chunk)
                for line in chunk.splitlines():
                    e = json.loads(line)
                    if "start" in e:
                        running = e
                    elif running is not None and e.get("nodeid") == running["start"]:
                        running = None
                if running is not None and time.time() - running["time"] > timeout:
                    proc.kill()
                    proc.wait()
                    timed_out = True
        _, running = outcomes(jsonl)
        if running is None and proc.returncode == RESTART:
            continue
        if running is None:
            scratch.unlink(missing_ok=True)
            if only_file:
                Path(only_file).unlink(missing_ok=True)
            return proc.returncode
        outcome = "timeout" if timed_out else "crashed"
        message = f"over {timeout} s" if timed_out else f"process exit code {proc.returncode}"
        with open(jsonl, "a", encoding="utf-8") as out:
            out.write(json.dumps(dict(nodeid=running["start"], outcome=outcome, message=message)) + "\n")


def result_paths(out_dir, test_file):
    stem = out_dir / test_file.replace("/", "__").removesuffix(".py")
    return stem.with_suffix(".jsonl"), stem.with_suffix(".cpu.jsonl"), stem.with_suffix(".log")


def is_vulkan_failure(entry):
    return entry["outcome"] == "failed" and FALLBACK not in entry.get("message", "")


def probe_device(out_dir):
    """The device's name and the DEVICE_DTYPES it can't compute in, saved to <out>/device.json"""
    code = ("import json, torch, torchvulkan\n"
            "unsupported = []\n"
            f"for name in {list(DEVICE_DTYPES)!r}:\n"
            "    try: (torch.ones(4, dtype=getattr(torch, name), device='vulkan') + 1).cpu()\n"
            "    except RuntimeError: unsupported.append(name)\n"
            "print(json.dumps(dict(device=torchvulkan.get_device_name(), unsupported=unsupported)))")
    probe = subprocess.run([sys.executable, "-c", code], env={**os.environ, "TORCHVULKAN_STRICT": "1"},
                           capture_output=True, text=True, check=True)
    (out_dir / "device.json").write_text(probe.stdout)
    return json.loads(probe.stdout)


def read_device(out_dir):
    path = out_dir / "device.json"
    return json.loads(path.read_text()) if path.exists() else dict(device="?", unsupported=[])


def run(args):
    tests_dir = args.tests or default_tests_dir()
    if not tests_dir.is_dir():
        fetch(tests_dir)
    files = args.files or discover(tests_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    device = probe_device(args.out)
    print(f"{device['device']}; dtypes it can't compute in: {', '.join(device['unsupported']) or 'none'}")
    if sys.platform == "darwin":
        # a Mac whose display sleeps stalls the GPU mid-run
        subprocess.Popen(["caffeinate", "-dims", "-w", str(os.getpid())])

    env = {**os.environ, "PYTORCH_TESTING_DEVICE_ONLY_FOR": "privateuse1"}
    if args.strict:
        env["TORCHVULKAN_STRICT"] = "1"
    cpu_env = {**os.environ, "PYTORCH_TESTING_DEVICE_ONLY_FOR": "cpu"}

    for i, test_file in enumerate(files, 1):
        jsonl, cpu_jsonl, log = result_paths(args.out, test_file)
        start = time.time()
        code = run_pytest(test_file, tests_dir, jsonl, log, env, VULKAN_CLASSES, args.timeout, args.pytest_args)
        results, _ = outcomes(jsonl)
        if not results and code not in (0, 5):
            print(f"[{i}/{len(files)}] {test_file}: pytest exited with {code} before running a test, see {log}")
            with open(args.out / "collection_errors.txt", "a", encoding="utf-8") as out:
                out.write(f"{test_file}\t{code}\n")
            continue

        failed = [n for n, e in results.items() if is_vulkan_failure(e)]
        if failed:
            run_pytest(test_file, tests_dir, cpu_jsonl, log, cpu_env, "CPU", args.timeout, args.pytest_args,
                       only=[cpu_twin(n) for n in failed])
        results_cpu, _ = outcomes(cpu_jsonl)
        counts = collections.Counter(classify(e, results_cpu, device["unsupported"]) for e in results.values())
        print(f"[{i}/{len(files)}] {test_file}: " + ", ".join(f"{n} {k}" for k, n in counts.most_common())
              + f" ({(time.time() - start) / 60:.1f} min)", flush=True)
    summary(args)


# ---------------------------------------------------------------- summary

OUTCOMES = ("passed", "failed", "fallback", "crashed", "timeout", "unsupported dtype", "expected", "fails on CPU too",
            "skipped", "xfailed", "xpassed")


def names_unsupported_dtype(entry, unsupported):
    method = entry["nodeid"].rsplit("::", 1)[-1]
    for dtype in unsupported:
        for name in DEVICE_DTYPES[dtype]:
            if re.search(rf"\b{name}\b", entry.get("message", "")) or re.search(rf"_{name}(_|$|\[)", method):
                return True
    return False


def expected_reason(entry):
    for nodeid, message, reason in EXPECTED:
        if re.search(nodeid, entry["nodeid"]) and re.search(message, entry.get("message", "")):
            return reason
    return None


def classify(entry, results_cpu, unsupported):
    outcome = entry["outcome"]
    if outcome != "failed":
        return outcome
    if FALLBACK in entry.get("message", ""):
        return "unsupported dtype" if names_unsupported_dtype(entry, unsupported) else "fallback"
    if expected_reason(entry):
        return "expected"
    twin = results_cpu.get(cpu_twin(entry["nodeid"]))
    # an xfailed twin is a failure PyTorch marks as expected on the CPU
    if twin is not None and twin["outcome"] in ("failed", "crashed", "timeout", "xfailed"):
        return "fails on CPU too"
    return outcome


def fallback_op(message):
    m = re.search(r"operation: (aten::[\w.]+?)\.?\s", message)
    if m:
        return m.group(1)
    m = re.search(r"\[NOT IMPLEMENTED\]: (.*?) Set TORCHVULKAN_STRICT", message)
    return m.group(1) if m else message


def summary(args):
    device = read_device(args.out)
    rows, fallbacks, messages, failures = [], collections.Counter(), collections.Counter(), []
    expected = collections.Counter()
    for jsonl in sorted(args.out.glob("*.jsonl")):
        if jsonl.name.endswith(".cpu.jsonl"):
            continue
        results, _ = outcomes(jsonl)
        if not results:
            continue
        results_cpu, _ = outcomes(jsonl.with_name(jsonl.name.removesuffix(".jsonl") + ".cpu.jsonl"))
        counts = collections.Counter()
        for e in results.values():
            kind = classify(e, results_cpu, device["unsupported"])
            counts[kind] += 1
            if kind == "fallback":
                fallbacks[fallback_op(e["message"])] += 1
            elif kind == "expected":
                expected[expected_reason(e)] += 1
            elif kind in ("failed", "crashed", "timeout"):
                messages[re.sub(r"\d+(\.\d+)?", "N", e.get("message", ""))[:150]] += 1
                failures.append(f"{e['nodeid']}  [{kind}] {e.get('message', '')}")
        rows.append((jsonl.stem.replace("__", "/") + ".py", counts))

    total = sum((c for _, c in rows), collections.Counter())
    lines = [f"# PyTorch test files on vulkan ({device['device']})", "",
             "| file | tests | " + " | ".join(OUTCOMES) + " |", "|---|---:|" + "---:|" * len(OUTCOMES)]
    for name, c in rows + [("**total**", total)]:
        lines.append(f"| {name} | {sum(c.values())} | " + " | ".join(str(c[k]) for k in OUTCOMES) + " |")
    lines += ["", "`fallback`: failed under TORCHVULKAN_STRICT=1 because an op would run on the CPU.",
              f"`unsupported dtype`: a fallback in a dtype this device can't compute in "
              f"({', '.join(device['unsupported']) or 'none'}).",
              "`expected`: a failure no PrivateUse1 backend can avoid (EXPECTED in this script).",
              "`fails on CPU too`: failed on vulkan, and the same test fails (or is an expected failure) on the CPU.", ""]
    errors_file = args.out / "collection_errors.txt"
    if errors_file.exists():
        errors = dict(line.split("\t") for line in errors_file.read_text().splitlines() if line)
        errors = {f: c for f, c in errors.items() if f not in {name for name, _ in rows}}
        if errors:
            lines += ["## Files that failed before running a test", ""]
            lines += [f"- {f} (pytest exit {c}; see its .log)" for f, c in sorted(errors.items())] + [""]
    if expected:
        lines += ["## Expected failures (tests)", ""] + [f"- {n} {r}" for r, n in expected.most_common()] + [""]
    if fallbacks:
        lines += ["## CPU fallbacks (tests)", ""] + [f"- {n} {op}" for op, n in fallbacks.most_common()] + [""]
    if messages:
        lines += ["## Most common vulkan failures (tests)", ""]
        lines += [f"- {n} `{m}`" for m, n in messages.most_common(40)] + [""]

    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (args.out / "failures.txt").write_text("\n".join(sorted(failures)) + "\n", encoding="utf-8")
    print("\n".join(lines[:len(rows) + 9]))
    print(f"\n{args.out / 'summary.md'}, {args.out / 'failures.txt'}")


def list_files(args):
    tests_dir = args.tests or default_tests_dir()
    if not tests_dir.is_dir():
        fetch(tests_dir)
    env = {**os.environ, "PYTORCH_TESTING_DEVICE_ONLY_FOR": "privateuse1"}
    total = 0
    for test_file in args.files or discover(tests_dir):
        code = ("import sys, torchvulkan, pytest; "
                f"sys.exit(pytest.main(['--collect-only', '-q', '-p', 'no:cacheprovider', {test_file!r}]))")
        out = subprocess.run([sys.executable, "-c", code], cwd=tests_dir, env=env, capture_output=True, text=True)
        n = sum(1 for line in out.stdout.splitlines() if re.search(r"(PRIVATEUSE1|VULKAN)::", line))
        total += n
        print(f"{n:8d}  {test_file}" + ("" if out.returncode in (0, 5) else f"  (collection exit {out.returncode})"),
              flush=True)
    print(f"{total:8d}  total")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        worker(sys.argv[2:])
    argv, pytest_args = sys.argv[1:], []
    if "--" in argv:
        argv, pytest_args = argv[:argv.index("--")], argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "list", "run", "summary"))
    parser.add_argument("files", nargs="*", help="test files relative to PyTorch's test/ (default: all)")
    parser.add_argument("--tests", type=Path, help="PyTorch's test/ directory (default: fetched into the cache)")
    parser.add_argument("--out", type=Path, default=Path("pytorch-test-results"), help="results directory")
    parser.add_argument("--timeout", type=float, default=600, help="seconds before a test counts as hung")
    parser.add_argument("--no-strict", dest="strict", action="store_false", help="allow CPU fallbacks")
    args = parser.parse_args(argv)
    args.pytest_args = pytest_args
    args.out = args.out.resolve()
    args.tests = args.tests.resolve() if args.tests else None

    if args.command == "fetch":
        fetch(args.tests or default_tests_dir())
    elif args.command == "list":
        list_files(args)
    elif args.command == "run":
        run(args)
    else:
        summary(args)


if __name__ == "__main__":
    main()
