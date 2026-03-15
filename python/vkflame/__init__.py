"""
vkflame: Vulkan-native inference. Drop-in replacement for ROCm AI compute.
"""
import ctypes
import os
from pathlib import Path

__version__ = "0.1.0"

_lib_search = [
    # Build tree (relative to this file's location: python/vkflame/)
    Path(__file__).parent.parent.parent / "build" / "libvkflame_rt.so",
    Path(__file__).parent.parent.parent / "build" / "vkflame_rt.dll",
    Path(__file__).parent.parent.parent / "build" / "Release" / "vkflame_rt.dll",
    Path("/usr/local/lib/libvkflame_rt.so"),
]


def _load_runtime() -> ctypes.CDLL:
    for p in _lib_search:
        if p.exists():
            try:
                return ctypes.CDLL(str(p))
            except OSError:
                continue
    raise ImportError(
        "[vkflame] Could not find libvkflame_rt.so / vkflame_rt.dll.\n"
        "  Run: cmake -B build && cmake --build build"
    )


_rt = _load_runtime()

rc = _rt.vkflame_init()
if rc != 0:
    raise ImportError(
        f"[vkflame] vkflame_init() failed (rc={rc}).\n"
        "  Check Vulkan GPU is present: vulkaninfo --summary"
    )

_rt.vkflame_get_context.restype = ctypes.c_void_p
_ctx = _rt.vkflame_get_context()


def info() -> None:
    """Print device name and supported features."""
    _rt.vkflame_print_info()


try:
    import sys as _sys
    _repo_root = str(Path(__file__).parent.parent.parent)
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    from torch_backend.mode import VKFlameMode, install  # noqa: F401
    # Note: CUDAPluggableAllocator cannot be registered after torch initialises
    # its CUDA/ROCm context (which happens on first tensor creation, before
    # vkflame is imported).  Zero-copy on Windows is handled instead by the
    # range-lookup in vkflame_buf_from_ptr + the download_into CPU-bounce path.
except ImportError:
    pass  # torch not installed — shim-only mode is still functional
