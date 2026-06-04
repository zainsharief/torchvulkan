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

def to_vulkan(obj):
    if isinstance(obj, torch.Tensor):    
        return obj.to('vulkan')
    elif isinstance(obj, str) and obj == 'cpu':
        return 'vulkan'
    elif isinstance(obj, torch.device) and obj.type == 'cpu':
        return torch.device('vulkan')
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_vulkan(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: to_vulkan(v) for k, v in obj.items()}
    return obj

def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.to('cpu')
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj

def is_not_implemented(exception: str):
    exception = exception.lower()

    if "not implemented" in exception:
        op_name = exception.split(' ')[-1]
        UNIMPLEMENTED_OPS[op_name] = UNIMPLEMENTED_OPS.get(op_name, 0) + 1
        return True
    
    elif "to be on cpu, but it's on vulkan" in exception.lower(): # for now, we just skip tests where values are on the wrong device
        return True
    
    return False

class TestVulkanOps(TestCase):
    
    @ops(op_db, allowed_dtypes=VULKAN_DTYPES)
    def test_correctness(self, device, dtype, op):

        if "as_strided_partial_views" in self._testMethodName:
            self.skipTest("Cross-device storage copy loses unreferenced base memory.")

        print(f"\nDEBUG: Attempting op '{op.name}' with dtype {dtype}: ", flush=True, end='')
        samples = op.sample_inputs(device, dtype)
        
        for sample in samples:
            cpu_input = sample.input
            cpu_args = sample.args
            cpu_kwargs = sample.kwargs
            expect_exeption = False

            try:
                expected = op(cpu_input, *cpu_args, **cpu_kwargs)
            except Exception:
                REMAINING_OPS.discard(op.name)
                expect_exeption = True
                
            try:
                vk_input = to_vulkan(cpu_input)
                vk_args = to_vulkan(cpu_args)
                vk_kwargs = to_vulkan(cpu_kwargs)

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

            elif op.name in ("pow", "__rpow__", "square", "float_power", "atan2") or dtype in (torch.float16, torch.bfloat16):
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

instantiate_device_type_tests(TestVulkanOps, globals(), only_for='cpu')

if __name__ == '__main__':
    run_tests()