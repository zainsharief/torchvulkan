import sys

try:
    import torch
except ImportError as e:
    raise ImportError("PyTorch is required to use torchvulkan. Please install PyTorch and try again.") from e

if torch.__version__ < "2.10.0":
    raise RuntimeError(f"torchvulkan [ERROR]: torchvulkan requires PyTorch 2.10.0 or higher, but found {torch.__version__}.")

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from . import _C

def is_available() -> bool:
    return _C.is_available()

def device_count() -> int:
    return _C.device_count()

def synchronize() -> None:
    _C.synchronize()

def empty_cache() -> None:
    _C.empty_cache()

torch.utils.rename_privateuse1_backend('vulkan')
torch._register_device_module('vulkan', sys.modules[__name__])