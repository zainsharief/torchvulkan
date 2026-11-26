"""No CPU fallback for what the device says it supports.

For every OpInfo op and every dtype that both the device reports (get_device_properties: has_fp64, has_int64,
has_fp16, has_bf16, has_int16, has_int8; float32 and int32 always) and the op supports on the CPU, a few sample
inputs run on the device in strict mode (TORCHVULKAN_STRICT=1), where every CPU fallback raises.

- The global fallback ("Silent fallback detected for operation: aten::X") means X has no Vulkan kernel: an op not
  implemented yet. The case is skipped and listed in the report, so the gaps stay visible without failing the run.
- Any other strict-mode message comes from a registered Vulkan kernel turning work away (a dtype or a shape it can't
  run, e.g. "A copy from Float to Half can't run"). That fails the test, unless the message names a dtype the device
  doesn't report (e.g. a copy to bfloat16 on a device without bfloat16), which is reported as kind 'unreported-dtype'.
- Other errors are not this suite's concern (test_pytorch_ops.py compares results and errors with the CPU).

Strict mode is read when the Vulkan context starts, so this file sets TORCHVULKAN_STRICT before importing
torchvulkan; run it in its own pytest process (it skips itself if the context was already up without strict mode).
The report (op, dtype, fallen-back aten op, registered or not, message) is written as JSON to
$TORCHVULKAN_FALLBACK_REPORT (default: torchvulkan-fallback-report.json in the current directory) and summarized at
the end of the run. TORCHVULKAN_FALLBACK_SAMPLES sets the samples per op and dtype (default 3).
"""
import json
import os
import re
import sys

# strict mode is fixed when the Vulkan context starts: if another module already started it without strict mode,
# setting the variable now changes nothing, and the cases are skipped
_STARTED_BEFORE = 'torchvulkan' in sys.modules and sys.modules['torchvulkan'].is_initialized()
_STRICT_BEFORE = os.environ.get('TORCHVULKAN_STRICT') == '1'
os.environ.setdefault('TORCHVULKAN_STRICT', '1')

import pytest  # noqa: E402
import torch  # noqa: E402
import torchvulkan  # noqa: E402
from torch.testing._internal.common_methods_invocations import op_db  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_map  # noqa: E402

DEVICE = 'vulkan'
SAMPLES = int(os.environ.get('TORCHVULKAN_FALLBACK_SAMPLES', '3'))
REPORT_PATH = os.environ.get('TORCHVULKAN_FALLBACK_REPORT', 'torchvulkan-fallback-report.json')
STRICT_ACTIVE = os.environ.get('TORCHVULKAN_STRICT') == '1' and (_STRICT_BEFORE or not _STARTED_BEFORE)


def reported_dtypes():
    p = torchvulkan.get_device_properties()
    dtypes = {torch.float32, torch.int32, torch.uint32, torch.bool}
    if p.has_fp64: dtypes.add(torch.float64)
    if p.has_int64: dtypes |= {torch.int64, torch.uint64}
    if p.has_fp16: dtypes.add(torch.float16)
    if p.has_bf16: dtypes.add(torch.bfloat16)
    if p.has_int16: dtypes |= {torch.int16, torch.uint16}
    if p.has_int8: dtypes |= {torch.int8, torch.uint8}
    return dtypes


DEVICE_DTYPES = reported_dtypes()
REPORT = []


def case_id(op):
    return op.name + ('.' + op.variant_test_name if op.variant_test_name else '')


def cases():
    out = []
    for op in op_db:
        for dtype in sorted(op.supported_dtypes('cpu') & DEVICE_DTYPES, key=str):
            out.append(pytest.param(op, dtype, id=f'{case_id(op)}-{str(dtype)[6:]}'))
    return out


class LastOp(TorchDispatchMode):
    """remembers the top-level aten op being dispatched, so a fallback without an op name can be attributed"""

    def __init__(self):
        super().__init__()
        self.current = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.current = func
        return func(*args, **(kwargs or {}))


def has_vulkan_kernel(qualname):
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(qualname, 'PrivateUse1')
    except RuntimeError:
        return False


# c10 ScalarType names as they appear in torchvulkan's messages
C10_DTYPES = {'Bool': torch.bool, 'Byte': torch.uint8, 'Char': torch.int8, 'Short': torch.int16, 'Int': torch.int32,
              'Long': torch.int64, 'UInt16': torch.uint16, 'UInt32': torch.uint32, 'UInt64': torch.uint64,
              'Half': torch.float16, 'BFloat16': torch.bfloat16, 'Float': torch.float32, 'Double': torch.float64,
              'ComplexHalf': torch.complex32, 'ComplexFloat': torch.complex64, 'ComplexDouble': torch.complex128,
              'Float8_e4m3fn': torch.float8_e4m3fn, 'Float8_e5m2': torch.float8_e5m2}


def classify(message, last):
    """(kind, aten op): kind is 'unimplemented', 'unreported-dtype' or 'registered-kernel-fallback' (a failure)"""
    m = re.search(r'operation:\s*(aten::[\w.]+?)\.?\s+Set', message)
    if m:
        name = m.group(1)
        if last is not None and last.name().split('.')[0] == name:
            name = last.name()
        return ('registered-kernel-fallback' if has_vulkan_kernel(name) else 'unimplemented'), name
    name = last.name() if last is not None else '?'
    named = {C10_DTYPES[w] for w in re.findall(r'\b(' + '|'.join(sorted(C10_DTYPES, key=len, reverse=True)) + r')\b', message)}
    if named - DEVICE_DTYPES:
        return 'unreported-dtype', name
    return 'registered-kernel-fallback', name


def to_device(x):
    return x.to(DEVICE) if isinstance(x, torch.Tensor) else x


@pytest.fixture(scope='session', autouse=True)
def write_report():
    yield
    with open(REPORT_PATH, 'w') as f:
        json.dump({'device': torchvulkan.get_device_properties().name, 'dtypes': sorted(str(d) for d in DEVICE_DTYPES),
                   'fallbacks': REPORT}, f, indent=1)
    reg = sorted({(r['aten_op'], r['dtype']) for r in REPORT if r['kind'] == 'registered-kernel-fallback'})
    unreg = sorted({r['aten_op'] for r in REPORT if r['kind'] == 'unimplemented'})
    unrep = sorted({r['aten_op'] for r in REPORT if r['kind'] == 'unreported-dtype'})
    out = sys.__stderr__
    print(f'\n[no-fallback] report: {REPORT_PATH}', file=out)
    print(f'[no-fallback] registered Vulkan kernels that fell back (failures): {len(reg)} op/dtype pairs', file=sys.__stderr__)
    for name, dt in reg[:60]:
        print(f'    {name} {dt}', file=sys.__stderr__)
    print(f'[no-fallback] aten ops without a Vulkan kernel (skipped): {len(unreg)}: ' + ', '.join(unreg[:80]), file=sys.__stderr__)
    print(f'[no-fallback] refused for a dtype the device does not report (skipped): ' + ', '.join(unrep[:40]), file=sys.__stderr__)


@pytest.mark.skipif(not STRICT_ACTIVE, reason='TORCHVULKAN_STRICT must be 1 before the Vulkan context starts')
@pytest.mark.parametrize('op,dtype', cases())
def test_no_cpu_fallback(op, dtype):
    samples = op.sample_inputs('cpu', dtype, requires_grad=False)
    registered_failures, unimplemented = [], []
    ran = 0
    for sample in samples:
        if ran >= SAMPLES:
            break
        ran += 1
        args = tree_map(to_device, (sample.input,) + tuple(sample.args))
        kwargs = tree_map(to_device, sample.kwargs)
        mode = LastOp()
        try:
            with mode:
                out = op(*args, **kwargs)
            torchvulkan.synchronize()
            del out
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if 'torchvulkan [NOT IMPLEMENTED]' not in msg:
                continue
            kind, name = classify(msg, mode.current)
            rec = {'op': case_id(op), 'dtype': str(dtype), 'aten_op': name, 'kind': kind,
                   'message': msg.split(' Set TORCHVULKAN_STRICT')[0][:300]}
            REPORT.append(rec)
            (registered_failures if kind == 'registered-kernel-fallback' else unimplemented).append(rec)
    if registered_failures:
        pytest.fail('CPU fallback from registered Vulkan kernels for a reported dtype:\n' +
                    '\n'.join(f"  {r['aten_op']} ({r['dtype']}): {r['message']}" for r in registered_failures), pytrace=False)
    if unimplemented:
        pytest.skip('; '.join(sorted({f"{r['kind']}: {r['aten_op']}" for r in unimplemented})))
    if ran == 0:
        pytest.skip('no sample inputs')
