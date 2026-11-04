import atexit
import sys
from collections import namedtuple

try:
    import torch
except ImportError as e:
    raise ImportError("PyTorch is required to use torchvulkan. Please install PyTorch and try again.") from e

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

try:
    from ._build_info import torch_version as _built_torch_version  # pyright: ignore[reportMissingImports]
except ImportError:
    _built_torch_version = None

def _torch_minor(version: str) -> tuple:
    return tuple(version.split("+")[0].split(".")[:2])

if _built_torch_version and _torch_minor(torch.__version__) != _torch_minor(_built_torch_version):
    _built = ".".join(_torch_minor(_built_torch_version))
    raise RuntimeError(
        f"torchvulkan [ERROR]: torchvulkan {__version__} was built for PyTorch {_built}.x, but found {torch.__version__}. "
        f"Install torch=={_built}.* or a torchvulkan release built for your PyTorch version."
    )

from . import _C

def is_available() -> bool:
    return _C.is_available()

def device_count() -> int:
    return _C.device_count()

def current_device() -> int:
    return _C.current_device()

def set_device(device) -> None:
    index = _device_index(device)
    if index >= 0:
        torch.accelerator.set_device_index(index)

def is_initialized() -> bool:
    return _C._is_initialized()

def _lazy_init() -> None:
    _C._lazy_init()

def device(dev):
    index = dev if isinstance(dev, int) or dev is None else torch.device(dev).index
    return torch.accelerator.device_index(index)

def synchronize(device=None) -> None:
    _C.synchronize(_device_index(device))

def empty_cache() -> None:
    _C.empty_cache()

def memory_allocated(device=None) -> int:
    return _C._memory_allocated(_device_index(device))

def max_memory_allocated(device=None) -> int:
    return _C._max_memory_allocated(_device_index(device))

def memory_reserved(device=None) -> int:
    return _C._memory_usage(_device_index(device))

def mem_get_info(device=None) -> tuple:
    return _C._mem_get_info(_device_index(device))

def reset_peak_memory_stats(device=None) -> None:
    _C._reset_peak_memory_stats(_device_index(device))

def memory_stats(device=None) -> dict:
    index = _device_index(device)
    allocated = _C._memory_allocated(index)
    peak = _C._max_memory_allocated(index)
    return {
        "active_bytes.all.current": allocated,
        "active_bytes.all.peak": peak,
        "allocated_bytes.all.current": allocated,
        "allocated_bytes.all.peak": peak,
        "reserved_bytes.all.current": _C._memory_usage(index),
    }

DeviceProperties = namedtuple("DeviceProperties", ["name", "total_memory", "vendor_id", "device_id", "api_version", "driver_version", "subgroup_size", "has_fp64", "has_int64", "has_fp16", "has_bf16", "has_int16", "has_int8"])

def get_device_properties(device=None) -> DeviceProperties:
    return DeviceProperties(**_C._device_properties(_device_index(device)))

def get_device_name(device=None) -> str:
    return get_device_properties(device).name

def _device_index(dev) -> int:
    if isinstance(dev, int):
        return dev
    index = None if dev is None else torch.device(dev).index
    return current_device() if index is None else index

def _default_generator(dev=None) -> torch.Generator:
    return _C._default_generator(_device_index(dev))

def _get_generator(device) -> torch.Generator:
    return _default_generator(device)

def __getattr__(name):
    if name == "default_generators":
        return tuple(_default_generator(index) for index in range(device_count()))
    if name == "_initialized":
        return is_initialized()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def _uint64_seed(seed) -> int:
    seed = int(seed)
    return seed + (1 << 64) if seed < 0 else seed

def manual_seed(seed) -> None:
    _C._manual_seed(current_device(), _uint64_seed(seed))

def manual_seed_all(seed) -> None:
    _C._manual_seed_all(_uint64_seed(seed))

def seed() -> None:
    _default_generator().seed()

def seed_all() -> None:
    random_seed = _default_generator(0).seed()
    for index in range(1, device_count()):
        _default_generator(index).manual_seed(random_seed)

def initial_seed() -> int:
    return _default_generator().initial_seed()

def get_rng_state(device="vulkan") -> torch.Tensor:
    return _default_generator(device).get_state()

def set_rng_state(new_state: torch.Tensor, device="vulkan") -> None:
    _default_generator(device).set_state(new_state.clone(memory_format=torch.contiguous_format))

def get_rng_state_all() -> list:
    return [get_rng_state(index) for index in range(device_count())]

def set_rng_state_all(new_states) -> None:
    for index, state in enumerate(new_states):
        set_rng_state(state, index)

def get_amp_supported_dtype() -> list:
    return _C._amp_supported_dtypes(current_device())

def is_bf16_supported() -> bool:
    return torch.bfloat16 in get_amp_supported_dtype()

def _is_in_bad_fork() -> bool:
    return _C._is_in_bad_fork()

atexit.register(_C._save_pipeline_caches)

torch.utils.rename_privateuse1_backend('vulkan')
torch._register_device_module('vulkan', sys.modules[__name__])

def _tensor_to_vulkan(self, device=None, non_blocking=False, **kwargs) -> torch.Tensor:
    return self.to(torch.device("vulkan", _device_index(device)), non_blocking=non_blocking, **kwargs)

# override torch.Tensor existing implementations for torch's own Vulkan backend
torch.Tensor.vulkan = _tensor_to_vulkan
torch.Tensor.is_vulkan = property(lambda self: self.device.type == "vulkan")
torch.utils.generate_methods_for_privateuse1_backend(for_tensor=False)
