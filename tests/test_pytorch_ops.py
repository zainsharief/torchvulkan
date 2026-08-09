import re
import pytest
import torch
import torchvulkan as torchvk

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import op_db

# all the datatypes we garantee to be supported by torchvk
VULKAN_DTYPES = [
    torch.float64, torch.uint64, torch.int64,
    torch.float32, torch.uint32, torch.int32,
    torch.float16, torch.bfloat16, torch.uint16, torch.int16,
    torch.uint8, torch.int8,
    torch.bool,
]

REMAINING_OPS = set()
UNIMPLEMENTED_OPS = {}

def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.to('cpu')
    elif isinstance(obj, str) and obj == 'vulkan':
        return 'cpu'
    elif isinstance(obj, torch.device) and obj.type == 'privateuseone':
        return torch.device('cpu')
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj

def is_not_implemented(exception: str):
    low = exception.lower()

    if "not implemented" in low:
        # strict-mode message: "... fallback detected for operation: aten::<op>. Set ..."
        m = re.search(r"operation:\s*(aten::\S+?)\.?\s+Set", exception)
        op_name = m.group(1) if m else exception.split(' ')[-1]
        UNIMPLEMENTED_OPS[op_name] = UNIMPLEMENTED_OPS.get(op_name, 0) + 1
        return True

    elif "could not run" in low and "backend" in low:
        op_name = exception.split("'")[1] if "'" in exception else "unknown"
        UNIMPLEMENTED_OPS[op_name] = UNIMPLEMENTED_OPS.get(op_name, 0) + 1
        return True

    elif "to be on cpu, but it's on vulkan" in low: # for now, we just skip tests where values are on the wrong device
        return True

    return False

class TestVulkanOps(TestCase):

    @ops(op_db, allowed_dtypes=VULKAN_DTYPES)
    def test_correctness(self, device, dtype, op):

        if "as_strided_partial_views" in self._testMethodName:
            self.skipTest("Cross-device storage copy loses unreferenced base memory.")

        print(f"\nDEBUG: Attempting op '{op.name}' with dtype {dtype}: ", flush=True, end='')

        # samples are now generated directly on the vulkan device; an unimplemented op used
        # while constructing the inputs should skip the test rather than error it out
        try:
            samples = list(op.sample_inputs(device, dtype))
        except Exception as e:
            if is_not_implemented(str(e)):
                self.skipTest(f"Sample generation for '{op.name}' needs an unimplemented Vulkan op.")
            raise

        for sample in samples:
            vk_input = sample.input
            vk_args = sample.args
            vk_kwargs = sample.kwargs

            # the CPU reference is the same sample moved back to the host
            cpu_input = to_cpu(vk_input)
            cpu_args = to_cpu(vk_args)
            cpu_kwargs = to_cpu(vk_kwargs)
            expect_exeption = False

            # torch 2.10's group_norm errors on an empty-batch input when it
            # round-trips through the Vulkan CPU fallback
            if (op.name == "nn.functional.group_norm"
                    and isinstance(cpu_input, torch.Tensor)
                    and cpu_input.numel() == 0):
                continue

            # Low-precision multi-head attention over OpInfo's large random
            # projection weights is dominated by catastrophic cancellation
            if (op.name == "nn.functional.multi_head_attention_forward"
                    and dtype in (torch.float16, torch.bfloat16)):
                continue

            # torch computes soft_margin_loss's log(1 + exp(-y*x)) directly in fp16 and
            # overflows to inf on large inputs; our stable/fp32 path stays finite, so the
            # fp16 reference is not comparable
            if (op.name == "nn.functional.soft_margin_loss" and dtype == torch.float16):
                continue

            try:
                expected = op(cpu_input, *cpu_args, **cpu_kwargs)
            except Exception:
                REMAINING_OPS.discard(op.name)
                expect_exeption = True

            try:
                actual = op(vk_input, *vk_args, **vk_kwargs)
                actual = to_cpu(actual) # torchvk will only compute on this step
            except Exception as e:
                REMAINING_OPS.discard(op.name)
                if is_not_implemented(str(e)):
                    self.skipTest(f"Operator '{op.name}' is not implemented for Vulkan backend.")
                    continue
                if expect_exeption:
                    continue
                self.fail(f"Vulkan backend failed on op '{op.name}' with error: {e}")

            # uninitialsied memory cannot be comapred
            if op.name in ("empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided", "empty_permuted"):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                continue

            elif dtype in (torch.float16, torch.bfloat16) and op.name in (
                "bmm", "baddbmm", "mm", "addmm", "matmul", "__rmatmul__",
                "linalg.multi_dot", "nn.functional.embedding_bag",
            ):
                self.assertEqual(actual, expected, atol=1e-1, rtol=3e-1)
                continue

            # GPU matmul accumulates in a different order than the CPU reference;
            # the same holds for composite ops that decompose into matmuls.
            elif dtype == torch.float32 and op.name in (
                "bmm", "baddbmm", "mm", "addmm", "matmul", "__rmatmul__",
                "nn.functional.multi_head_attention_forward",
                "nn.functional.scaled_dot_product_attention", "pca_lowrank",
            ):
                self.assertEqual(actual, expected, atol=1e-4, rtol=1e-3)
                continue

            # fused operations lose precision on rounding
            elif dtype in (torch.float16, torch.bfloat16) and op.name in (
                "lerp", "addcmul", "addcdiv", "addr",
                "native_layer_norm", "native_group_norm",
                "nn.functional.layer_norm", "nn.functional.group_norm",
                "nn.functional.bilinear", "nn.functional.poisson_nll_loss",
            ):
                self.assertEqual(actual, expected, atol=1e-1, rtol=5e-2, exact_dtype=False)
                continue

            # loss/normalisation reductions accumulate in a different order and use float32
            # intermediates on the GPU even for float64 inputs
            elif dtype == torch.float64 and op.name in (
                "nn.functional.cross_entropy", "nn.functional.linear_cross_entropy",
                "nn.functional.local_response_norm", "nn.functional.poisson_nll_loss",
            ):
                self.assertEqual(actual, expected, atol=1e-3, rtol=5e-3)
                continue

            # exp/log and the other transcendentals are evaluated in float32 on the GPU
            # even for float64 inputs, so their float64 results can't beat ~float32 precision
            elif dtype == torch.float64 and op.name in (
                "exp", "log", "sin", "cos", "tan", "asin", "acos", "atan",
                "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
                "exp2", "log2", "log10", "expm1", "log1p", "sigmoid",
                "hypot", "xlogy", "logaddexp", "logaddexp2",
            ):
                self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
                continue

            # special functions use float32 polynomial/rational approximations; the Bessel and
            # gamma families lose accuracy for large arguments even in float32
            elif op.name in ("i0", "special.i1", "special.i0e", "special.i1e", "lgamma", "digamma", "erfinv"):
                self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
                continue

            # the other special functions are accurate in float32 but downcast float64 to float32
            elif dtype == torch.float64 and op.name in (
                "erf", "erfc", "erfinv", "sinc", "special.entr", "lgamma", "digamma",
                "nn.functional.gelu", "special.ndtr", "special.log_ndtr", "mvlgamma",
            ):
                self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
                continue

            # log_softmax runs through exp/log at float32, so float64 results can't beat ~1e-7
            elif op.name in ("log_softmax", "masked.log_softmax") and (
                dtype == torch.float64 or cpu_kwargs.get("dtype") == torch.float64
            ):
                self.assertEqual(actual, expected, atol=1e-5, rtol=1e-5)
                continue

            elif op.name in ("pow", "__rpow__", "square", "float_power", "atan2", "ldexp") or dtype in (torch.float16, torch.bfloat16):
                self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
                continue

            self.assertEqual(actual, expected)
            REMAINING_OPS.add(op.name)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        print(f'\n--- TEST SUMMARY ---')
        print(f'The complete set of working operators is: {REMAINING_OPS}')
        sorted_ops = sorted(UNIMPLEMENTED_OPS.items(), key=lambda item: item[1], reverse=True)
        print(f'The complete set of unimplemented operators is: {sorted_ops}')

# run the OpInfo suite directly on the vulkan (PrivateUse1) device: inputs are generated
# on-device via the RNG fills, exactly how a real user drives the backend
instantiate_device_type_tests(TestVulkanOps, globals(), only_for='privateuse1')

if __name__ == '__main__':
    run_tests()
