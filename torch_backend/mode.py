"""
vkflame TorchDispatchMode — intercepts PyTorch aten ops and routes them
to the Vulkan compute backend via ctypes.
"""
import os
import sys
import warnings
from typing import Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_DEBUG = os.environ.get("VKFLAME_DEBUG") == "1"

# Cached aten op reference for GQA fallthrough check
try:
    _SDPA_OP = torch.ops.aten.scaled_dot_product_attention.default
except AttributeError:
    _SDPA_OP = None


class VKFlameMode(TorchDispatchMode):
    """Drop the mode into PyTorch's dispatch stack to intercept aten ops."""

    _HANDLED_OPS: dict  # populated at bottom of file

    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        handler = self._HANDLED_OPS.get(op)
        if handler is None:
            return op(*args, **kwargs)
        try:
            if _DEBUG:
                print(f"[vkflame] {op}", file=sys.stderr)
            return handler(*args, **kwargs)
        except Exception as e:
            warnings.warn(f"[vkflame] {op} failed, falling through to PyTorch: {e}")
            # GQA-aware SDPA fallthrough: PyTorch < 2.2 native SDPA doesn't
            # support Hq≠Hkv without explicit K/V expansion.
            if _SDPA_OP is not None and op is _SDPA_OP and len(args) >= 3:
                q, k, v = args[0], args[1], args[2]
                if k.dim() == 4 and q.dim() == 4 and k.shape[1] != q.shape[1]:
                    rep = q.shape[1] // k.shape[1]
                    k_exp = k.repeat_interleave(rep, dim=1)
                    v_exp = v.repeat_interleave(rep, dim=1)
                    return op(q, k_exp, v_exp, *args[3:], **kwargs)
            return op(*args, **kwargs)


_mode_instance: Optional[VKFlameMode] = None

_orig_sdpa = None
# Lazily determined: does the installed torch F.sdpa accept a `scale` kwarg?
_sdpa_accepts_scale: Optional[bool] = None


def _gqa_aware_sdpa(
    query, key, value,
    attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs
):
    """Drop-in for F.scaled_dot_product_attention that handles GQA by expanding
    KV heads *before* PyTorch's own pre-dispatch shape check fires.

    ``scale`` is forwarded as a keyword argument so it works on both the
    6-positional-arg (older) and 7-positional-arg (newer) C bindings.
    """
    global _sdpa_accepts_scale
    if (
        query.dim() == 4 and key.dim() == 4
        and query.shape[1] != key.shape[1]
        and query.shape[1] % key.shape[1] == 0
    ):
        rep = query.shape[1] // key.shape[1]
        key   = key.repeat_interleave(rep, dim=1)
        value = value.repeat_interleave(rep, dim=1)
    # Probe once: inspect.signature fails on C builtins, so use a live test call.
    if _sdpa_accepts_scale is None:
        try:
            _orig_sdpa(
                torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1),
                torch.zeros(1, 1, 1, 1), scale=1.0,
            )
            _sdpa_accepts_scale = True
        except TypeError:
            _sdpa_accepts_scale = False
    if _sdpa_accepts_scale:
        return _orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal,
                          scale=scale, **kwargs)
    return _orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, **kwargs)


def install(permanent: bool = False):
    """
    Enable the vkflame dispatch mode.

    Parameters
    ----------
    permanent : bool
        If True the mode is activated globally for the lifetime of the
        process.  If False a context-manager is returned so the caller
        can use it with ``with vkflame.install():``.
    """
    global _mode_instance, _orig_sdpa
    # Patch F.sdpa so GQA (Hq != Hkv) is expanded before PyTorch's shape check
    import torch.nn.functional as _F
    if _orig_sdpa is None:
        _orig_sdpa = _F.scaled_dot_product_attention
        _F.scaled_dot_product_attention = _gqa_aware_sdpa

    if permanent:
        _mode_instance = VKFlameMode()
        _mode_instance.__enter__()
        return _mode_instance
    else:
        return VKFlameMode()


# ── Import op handlers — must happen after class definition ─────────
# The ops module attaches the map to VKFlameMode._HANDLED_OPS.
try:
    from . import ops as _ops_module  # noqa: F401 — side-effectful import
    VKFlameMode._HANDLED_OPS = _ops_module.build_handler_map()
except Exception as _e:
    warnings.warn(f"[vkflame] Could not load op handlers: {_e}")
    VKFlameMode._HANDLED_OPS = {}
