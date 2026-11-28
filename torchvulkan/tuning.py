"""The autotuner's cached results.

torchvulkan times kernel configurations, and the ways of running some ops (a direct kernel, a composite of other
ops, another dtype through casts), on first use for each shape. Results live in memory and in a tuning file in the
cache directory, so later processes reuse them. These functions re-time or forget them on demand::

    with torchvulkan.tuning.retune():   # time again what this block uses, whatever is cached, and store it
        model(example)

    torchvulkan.tuning.clear()                 # forget this process's results; each is timed again on next use
    torchvulkan.tuning.clear(persistent=True)  # and empty the tuning file

TORCHVULKAN_RETUNE=1 retunes a whole process, TORCHVULKAN_TUNE_LOG=1 prints what is timed and chosen.
"""
import contextlib

import torchvulkan


def _index(device):
    return torchvulkan._device_index(device)


def clear(persistent=False, device=None) -> None:
    """Forget every tuned result of this process so each is timed again on its next use.

    With ``persistent`` the tuning file is emptied too; otherwise it keeps its lines and new timings are appended
    (a later line wins). Don't call it while ops run on the device from another thread.
    """
    torchvulkan._C._tuning_clear(_index(device), bool(persistent))


@contextlib.contextmanager
def retune(device=None):
    """Within the block, time each kernel configuration and op path again on its first use, then store it."""
    index = _index(device)
    torchvulkan._C._tuning_retune(index, True)
    try:
        yield
    finally:
        torchvulkan.synchronize(index)
        torchvulkan._C._tuning_retune(index, False)


def entries(device=None) -> int:
    """How many tuned results this process knows, from the file and timed since."""
    return int(torchvulkan._C._tuning_entries(_index(device)))


def cache_file(device=None) -> str:
    """The tuning file, or an empty string when the cache is off."""
    return torchvulkan._C._tuning_cache_file(_index(device))
