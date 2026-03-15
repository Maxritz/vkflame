"""
vkflame op handlers — implements each intercepted aten op by calling the
C++ runtime via ctypes.

Works on both Linux/ROCm (CUDA tensors) and Windows (CPU tensors via staged
Vulkan uploads).  Float32 tensors always fall through to PyTorch so that
reference computations in tests are unaffected.

Call ``build_handler_map()`` to get the ``{op: callable}`` dict that
VKFlameMode uses.
"""
from __future__ import annotations

import ctypes
import math
import os
import platform
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import torch

_DEBUG = os.environ.get("VKFLAME_DEBUG") == "1"
_WINDOWS = platform.system() == "Windows"

# On Windows, torch.Tensor.is_cuda == True means ROCm/HIP memory.
# HIP device pointers live in a different virtual address space than Vulkan
# device addresses, so passing them directly to vkflame_dispatch_* produces
# garbage.  Always use the staged CPU↔Vulkan upload/download path on Windows.
_DIRECT_GPU_OK = not _WINDOWS

# ── Dtypes we actively accelerate (fp16, int8, bf16).
# fp32 always falls through so reference computations are unaffected.
_ACCEL_DTYPES = {torch.float16, torch.int8, torch.bfloat16}

# ── Load C runtime ─────────────────────────────────────────────────
def _load_rt() -> ctypes.CDLL:
    # ops.py lives at <repo>/torch_backend/ops.py
    # DLL lives at    <repo>/build/Release/vkflame_rt.dll
    _repo = Path(__file__).parent.parent
    candidates = [
        _repo / "build" / "libvkflame_rt.so",
        _repo / "build" / "vkflame_rt.dll",
        _repo / "build" / "Release" / "vkflame_rt.dll",
        Path("/usr/local/lib/libvkflame_rt.so"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return ctypes.CDLL(str(p))
            except OSError:
                continue
    raise ImportError(
        "[vkflame] Could not find vkflame_rt shared library. "
        "Run: cmake -B build && cmake --build build"
    )


_rt: Optional[ctypes.CDLL] = None
_ctx: Optional[ctypes.c_void_p] = None


def _ensure_runtime() -> tuple[ctypes.CDLL, ctypes.c_void_p]:
    global _rt, _ctx
    if _rt is None:
        _rt = _load_rt()
        rc = _rt.vkflame_init()
        if rc != 0:
            raise RuntimeError(f"[vkflame] vkflame_init() failed (rc={rc})")
        _rt.vkflame_get_context.restype = ctypes.c_void_p
        _rt.vkflame_alloc.restype = ctypes.c_void_p
        _rt.vkflame_buf_address.restype = ctypes.c_uint64
        # Win32 zero-copy helpers
        if _WINDOWS:
            if hasattr(_rt, 'vkflame_wrap_hip_ptr'):
                _rt.vkflame_wrap_hip_ptr.restype = ctypes.c_void_p
                _rt.vkflame_wrap_hip_ptr.argtypes = [
                    ctypes.c_void_p, ctypes.c_size_t]
            if hasattr(_rt, 'vkflame_buf_from_hip_ptr'):
                _rt.vkflame_buf_from_hip_ptr.restype = ctypes.c_uint64
                _rt.vkflame_buf_from_hip_ptr.argtypes = [ctypes.c_uint64]
        _ctx = ctypes.c_void_p(_rt.vkflame_get_context())
    return _rt, _ctx


# ── Dtype helpers ──────────────────────────────────────────────────
_DTYPE_MAP = {
    torch.float32:  0,
    torch.float16:  1,
    torch.int8:     2,
    torch.bfloat16: 3,
}


def _dtype_to_int(dt: torch.dtype) -> int:
    return _DTYPE_MAP.get(dt, 0)


# ── Windows staged-upload helper ───────────────────────────────────
class _VulkanBuf:
    """Wrap a single Vulkan device buffer for staged CPU→GPU→CPU dispatch."""

    __slots__ = ("_rt", "_buf_ptr", "_addr", "_nbytes")

    def __init__(self, rt: ctypes.CDLL, nbytes: int) -> None:
        self._rt = rt
        self._buf_ptr = ctypes.c_void_p(int(rt.vkflame_alloc(nbytes)))
        if not self._buf_ptr.value:
            raise MemoryError("[vkflame] vkflame_alloc failed")
        self._addr = ctypes.c_void_p(int(rt.vkflame_buf_address(self._buf_ptr)))
        self._nbytes = nbytes

    @property
    def as_arg(self) -> ctypes.c_void_p:
        """Device address cast to void* — pass directly to vkflame_dispatch_*."""
        return self._addr

    def upload(self, t: torch.Tensor) -> None:
        cont = t.contiguous()
        if cont.is_cuda:
            cont = cont.cpu()
        self._rt.vkflame_memcpy_h2d(
            self._buf_ptr, ctypes.c_void_p(cont.data_ptr()), self._nbytes, 0)

    def download_into(self, t: torch.Tensor) -> None:
        if t.is_cuda:
            # On Windows, t.data_ptr() is a HIP GPU virtual address — not
            # CPU-accessible. Download to a pinned CPU buffer first, then
            # copy to the GPU tensor via HIP.
            cpu = torch.empty(t.numel(), dtype=t.dtype, device='cpu')
            self._rt.vkflame_memcpy_d2h(
                ctypes.c_void_p(cpu.data_ptr()), self._buf_ptr, self._nbytes, 0)
            t.copy_(cpu.view_as(t))
        else:
            self._rt.vkflame_memcpy_d2h(
                ctypes.c_void_p(t.data_ptr()), self._buf_ptr, self._nbytes, 0)

    def free(self) -> None:
        if self._buf_ptr and self._buf_ptr.value:
            self._rt.vkflame_free(self._buf_ptr)
            self._buf_ptr = ctypes.c_void_p(0)

    def __del__(self) -> None:
        self.free()


def _is_gpu(t: Optional[torch.Tensor]) -> bool:
    """True when this tensor can be passed to Vulkan dispatch without staging.

    Linux (ROCm): Vulkan and HIP share the same VA — data_ptr() is usable directly.
    Windows: only true if the tensor was allocated via our custom allocator and
    has a registered zero-copy VkBuffer (checked via vkflame_buf_from_hip_ptr).
    """
    if t is None:
        return False
    if not t.is_cuda:
        return False
    if _DIRECT_GPU_OK:  # Linux: all CUDA tensors are directly usable
        return True
    # Windows: check if a zero-copy VkBuffer was registered for this HIP address
    rt, _ = _ensure_runtime()
    if not hasattr(rt, 'vkflame_buf_from_hip_ptr'):
        return False
    return bool(rt.vkflame_buf_from_hip_ptr(ctypes.c_uint64(t.data_ptr())))


@contextmanager
def _staged(
    rt: ctypes.CDLL,
    inputs: list[Optional[torch.Tensor]],
    output: torch.Tensor,
) -> Generator:
    """
    Yield (in_args, out_arg) ready for a vkflame_dispatch_* call.

    Priority order:
      1. Linux/ROCm direct:     CUDA tensors → data_ptr() straight to Vulkan
                                (HIP and Vulkan share the same VA on Linux).
      2. Windows zero-copy:     CUDA (HIP) tensors exported via KMT handle and
                                imported as VkBuffer — no staging memcpy.
                                Requires VK_KHR_external_memory_win32 + ROCm 5.4+.
                                Falls through to (3) if any wrap fails.
      3. Staged (fallback):     CPU tensors or failed zero-copy → upload inputs,
                                dispatch, download output.  Works everywhere,
                                but has significant PCIe overhead on Windows.
    """
    # ── (1) Linux direct path ──────────────────────────────────────────────
    use_direct = all(_is_gpu(t) for t in inputs if t is not None) and _is_gpu(output)
    if use_direct and not _WINDOWS:
        in_args = [
            ctypes.c_void_p(t.data_ptr()) if t is not None else ctypes.c_void_p(0)
            for t in inputs
        ]
        yield in_args, ctypes.c_void_p(output.data_ptr())
        return

    # ── (2) Windows zero-copy via VK_KHR_external_memory_win32 ────────────
    # Enabled when all tensors were allocated via vkflame_pytorch_malloc  
    # (registered in g_hip_ptr_map) — those are top-level WDDM resources
    # that hipMemGetExportHandle can process without triggering device lost.
    _all_gpu = (all(t is None or _is_gpu(t) for t in inputs) and _is_gpu(output))
    if (_WINDOWS
            and _all_gpu
            and hasattr(rt, 'vkflame_buf_from_hip_ptr')):
        # All inputs/output have registered VkBuffers — use their device addresses directly.
        # vkflame_buf_from_hip_ptr(hip_ptr) → Vulkan device address (uint64)
        try:
            in_args = []
            for t in inputs:
                if t is None:
                    in_args.append(ctypes.c_void_p(0))
                else:
                    vk_addr = rt.vkflame_buf_from_hip_ptr(ctypes.c_uint64(t.data_ptr()))
                    in_args.append(ctypes.c_void_p(int(vk_addr)))
            out_vk_addr = rt.vkflame_buf_from_hip_ptr(ctypes.c_uint64(output.data_ptr()))
            yield in_args, ctypes.c_void_p(int(out_vk_addr))
            return
        except Exception as _e:
            if _DEBUG:
                import sys
                print(f"[vkflame] Win32 zero-copy lookup failed: {_e}, falling to staging",
                      file=sys.stderr)

    # ── (3) Staged fallback ────────────────────────────────────────────────
    in_bufs: list[Optional[_VulkanBuf]] = []
    for t in inputs:
        if t is None:
            in_bufs.append(None)
        else:
            b = _VulkanBuf(rt, t.nbytes)
            b.upload(t)
            in_bufs.append(b)
    out_buf = _VulkanBuf(rt, output.nbytes)
    try:
        in_args = [b.as_arg if b is not None else ctypes.c_void_p(0) for b in in_bufs]
        yield in_args, out_buf.as_arg
        out_buf.download_into(output)
    finally:
        for b in in_bufs:
            if b is not None:
                b.free()
        out_buf.free()


# ── Op handlers ───────────────────────────────────────────────────

def _mm_handler(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dtype not in _ACCEL_DTYPES:
        raise NotImplementedError  # let fp32 fall through for reference computations
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    rt, ctx = _ensure_runtime()
    out = torch.empty(M, N, dtype=a.dtype, device=a.device)
    with _staged(rt, [a, b, None], out) as (in_args, out_arg):
        rt.vkflame_dispatch_linear(
            ctx, in_args[0], in_args[1], in_args[2], out_arg,
            M, N, K, _dtype_to_int(a.dtype),
            0, 0, 0, None, None, None,
        )
    return out


def _addmm_handler(
    bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    beta: float = 1.0, alpha: float = 1.0,
) -> torch.Tensor:
    if a.dtype not in _ACCEL_DTYPES:
        raise NotImplementedError
    result = _mm_handler(a, b)
    if beta != 0.0 and bias is not None:
        result = result + bias * beta
    if alpha != 1.0:
        result = result * alpha
    return result


def _linear_handler(
    x: torch.Tensor, weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x.dtype not in _ACCEL_DTYPES:
        raise NotImplementedError
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    out_2d = _mm_handler(x_2d, weight.t())
    out = out_2d.reshape(*orig_shape[:-1], weight.shape[0])
    if bias is not None:
        out = out + bias
    return out


def _sdpa_handler(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if query.dtype not in _ACCEL_DTYPES:
        raise NotImplementedError
    B, Hq, Sq, D = query.shape
    _B, Hkv, Skv, _D = key.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Try Vulkan flash-attention (handles GQA natively in shader)
    try:
        rt, ctx = _ensure_runtime()
        out = torch.empty_like(query)
        with _staged(rt, [query, key, value], out) as (in_args, out_arg):
            rt.vkflame_dispatch_flash_attention(
                ctx, in_args[0], in_args[1], in_args[2], out_arg,
                B, Hq, Hkv, Sq, Skv, D,
                ctypes.c_float(scale), int(is_causal),
            )
        return out
    except Exception:
        pass  # fall through to Python reference below

    # Python fallback (Vulkan unavailable or GQA not supported on this config).
    # Expand KV heads to match Q so standard batched matmul works.
    k_use = key.repeat_interleave(Hq // Hkv, dim=-3) if (Hkv != Hq and Hq % Hkv == 0) else key
    v_use = value.repeat_interleave(Hq // Hkv, dim=-3) if (Hkv != Hq and Hq % Hkv == 0) else value
    # Compute attention in fp32 using ops not in handler map (bmm/amax/exp/sum → passthrough)
    q_f = query.float()
    k_f = k_use.float()
    v_f = v_use.float()
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    if is_causal:
        mask = torch.ones(Sq, Skv, dtype=torch.bool, device=query.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    sf = scores - scores.amax(dim=-1, keepdim=True)
    sf = torch.exp(sf)
    sf = sf / sf.sum(dim=-1, keepdim=True)
    return torch.matmul(sf, v_f).to(query.dtype)


def _rms_norm_handler(
    x: torch.Tensor,
    normalized_shape: list,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> tuple:
    if x.dtype not in _ACCEL_DTYPES or bias is not None:
        raise NotImplementedError
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    rt, ctx = _ensure_runtime()
    out = torch.empty_like(x)
    gamma = weight if weight is not None else torch.ones(N, dtype=x.dtype, device=x.device)
    with _staged(rt, [x, gamma], out) as (in_args, out_arg):
        rt.vkflame_dispatch_rms_norm(
            ctx, in_args[0], in_args[1], out_arg,
            M, N, ctypes.c_float(eps),
        )
    return out, torch.zeros(M, dtype=torch.float32, device=x.device), \
                torch.ones(M,  dtype=torch.float32, device=x.device)


def _softmax_handler(x: torch.Tensor, dim: int = -1, half_to_float: bool = False) -> torch.Tensor:
    if x.dtype not in _ACCEL_DTYPES:
        raise NotImplementedError
    if dim != -1 and dim != len(x.shape) - 1:
        raise NotImplementedError(f"softmax only on last dim, got dim={dim}")
    # Vulkan softmax_online.glsl uses fp32 storage — convert if needed so the
    # shader reads and writes the correct number of elements.
    x_f32 = x.to(torch.float32) if x.dtype != torch.float32 else x
    M = x_f32.numel() // x_f32.shape[-1]
    N = x_f32.shape[-1]
    rt, ctx = _ensure_runtime()
    out_f32 = torch.empty_like(x_f32)
    with _staged(rt, [x_f32], out_f32) as (in_args, out_arg):
        rt.vkflame_dispatch_softmax(ctx, in_args[0], out_arg, M, N)
    # If caller expects fp16 (half_to_float=False) cast back; if True keep fp32
    return out_f32 if half_to_float or x.dtype == torch.float32 else out_f32.to(x.dtype)


def _softmax_aten_handler(x: torch.Tensor, dim: int, half_to_float: bool) -> torch.Tensor:
    return _softmax_handler(x, dim, half_to_float)


def _topk_handler(
    x: torch.Tensor, k: int, dim: int = -1,
    largest: bool = True, sorted: bool = True,
) -> tuple:
    if dim != -1 and dim != len(x.shape) - 1:
        raise NotImplementedError(f"topk only on last dim, got dim={dim}")
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    rt, ctx = _ensure_runtime()
    vals = torch.empty(*x.shape[:-1], k, dtype=x.dtype, device=x.device)
    idxs = torch.empty(*x.shape[:-1], k, dtype=torch.int32, device=x.device)
    # topk needs vals and idxs as paired outputs — stage both alongside input
    with _staged(rt, [x], vals) as (in_args, vals_arg):
        idxs_buf: Optional[_VulkanBuf] = None
        if not _is_gpu(x):
            idxs_buf = _VulkanBuf(rt, idxs.nbytes)
        try:
            idxs_arg = idxs_buf.as_arg if idxs_buf else ctypes.c_void_p(idxs.data_ptr())
            rt.vkflame_dispatch_topk(
                ctx, in_args[0], vals_arg, idxs_arg,
                M, N, k, int(largest),
            )
            if idxs_buf:
                idxs_buf.download_into(idxs)
        finally:
            if idxs_buf:
                idxs_buf.free()
    return vals, idxs.to(torch.int64)


def _argmax_handler(x: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
    if dim is None:
        x_flat = x.reshape(1, -1)
        _, idx = _topk_handler(x_flat, 1, dim=-1, largest=True)
        return idx.reshape(()) if not keepdim else idx.reshape(1)
    _, idx = _topk_handler(x, 1, dim=dim, largest=True)
    if not keepdim:
        idx = idx.squeeze(dim)
    return idx


def _embedding_handler(
    weight: torch.Tensor, indices: torch.Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    if weight.dtype not in _ACCEL_DTYPES:
        raise NotImplementedError
    V, D = weight.shape
    B = indices.numel()
    rt, ctx = _ensure_runtime()
    idx_i32 = indices.reshape(-1).to(torch.int32).contiguous()
    out = torch.empty(B, D, dtype=weight.dtype, device=weight.device)
    with _staged(rt, [weight, idx_i32], out) as (in_args, out_arg):
        rt.vkflame_dispatch_embedding(ctx, in_args[0], in_args[1], out_arg, V, D, B)
    return out.reshape(*indices.shape, D)


def _multinomial_handler(
    input: torch.Tensor,
    num_samples: int,
    replacement: bool = False,
    generator=None,
) -> torch.Tensor:
    # Sampling is not latency-critical; delegate to PyTorch for correctness
    return torch.multinomial(input.cpu().float(), num_samples, replacement,
                             generator=generator).to(input.device)


def _cumsum_handler(x: torch.Tensor, dim: int, dtype=None) -> torch.Tensor:
    raise NotImplementedError("cumsum fallthrough")


def _sort_handler(x: torch.Tensor, dim: int = -1, descending: bool = False, stable: bool = False):
    raise NotImplementedError("sort fallthrough")


def _activation_handler(x: torch.Tensor, act_id: int) -> torch.Tensor:
    if x.dtype not in _ACCEL_DTYPES:
        raise NotImplementedError
    rt, ctx = _ensure_runtime()
    out = torch.empty_like(x)
    M = x.numel()
    # B must be a 1-element tensor of 1.0 so that A[i]*B[0] = A[i]*1 = A[i].
    # Passing B=x was wrong: the linear shader reads B[0,0]=x[0], giving act(x[i]*x[0]).
    ones = torch.ones(1, dtype=x.dtype, device=x.device)
    with _staged(rt, [x, ones], out) as (in_args, out_arg):
        rt.vkflame_dispatch_linear(
            ctx, in_args[0], in_args[1], None, out_arg,
            M, 1, 1, _dtype_to_int(x.dtype), 0, 0, act_id, None, None, None,
        )
    return out


def _relu_handler(x: torch.Tensor) -> torch.Tensor:
    return _activation_handler(x, 2)


def _silu_handler(x: torch.Tensor) -> torch.Tensor:
    return _activation_handler(x, 1)


def _gelu_handler(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    return _activation_handler(x, 3)


# Binary op codes — must match VKF_BINOP_* in dispatch.h
_BINOP_ADD = 0
_BINOP_MUL = 1
_BINOP_SUB = 2
_BINOP_DIV = 3


def _binop_handler(
    a: torch.Tensor, b: torch.Tensor, op_id: int, alpha: float = 1.0
) -> torch.Tensor:
    """GPU element-wise binary op via binop_f32.glsl; handles broadcasting."""
    if a.dtype not in _ACCEL_DTYPES:
        raise NotImplementedError
    # b may be a Python scalar (e.g. aten.add.Tensor called with alpha=0.1)
    if not isinstance(b, torch.Tensor):
        b = torch.full_like(a, float(b))
    # binop_f32.glsl is fp32-only; convert inputs and restore original dtype on output
    b_scaled = b if alpha == 1.0 else b * alpha
    a_f32 = a.to(torch.float32).contiguous()
    b_f32 = b_scaled.to(torch.float32).contiguous()
    # Expand to common broadcast shape so both buffers have identical element counts
    out_shape = torch.broadcast_shapes(a_f32.shape, b_f32.shape)
    a_bc = a_f32.expand(out_shape).contiguous()
    b_bc = b_f32.expand(out_shape).contiguous()
    n = a_bc.numel()
    rt, ctx = _ensure_runtime()
    out_f32 = torch.empty(out_shape, dtype=torch.float32, device=a.device)
    with _staged(rt, [a_bc, b_bc], out_f32) as (in_args, out_arg):
        rt.vkflame_dispatch_binop_f32(ctx, in_args[0], in_args[1], out_arg, n, n, op_id)
    return out_f32.to(a.dtype)


# ── Handler map ────────────────────────────────────────────────────

def build_handler_map() -> dict:
    """Return the {op → handler} dict for VKFlameMode.

    Each entry is registered with a try-except so a missing overload in the
    installed PyTorch version silently skips that entry rather than aborting
    the whole map.
    """
    aten = torch.ops.aten

    # (attribute_path, handler) pairs
    _raw: list[tuple[str, object]] = [
        ("mm.default",                              _mm_handler),
        ("addmm.default",                           _addmm_handler),
        ("linear.default",                          _linear_handler),
        ("scaled_dot_product_attention.default",    _sdpa_handler),
        ("native_layer_norm.default",               _rms_norm_handler),
        # softmax: both overloads — _softmax.default (low-level) + softmax.Tensor (F.softmax)
        ("_softmax.default",                        _softmax_aten_handler),
        ("softmax.Tensor",   lambda x, dim, dtype=None: _softmax_handler(x, dim)),
        ("relu.default",                            _relu_handler),
        ("silu.default",                            _silu_handler),
        ("gelu.default",                            _gelu_handler),
        ("embedding.default",                       _embedding_handler),
        ("topk.default",                            _topk_handler),
        ("argmax.default",                          _argmax_handler),
        ("multinomial.replacement",                 _multinomial_handler),
        # cumsum / sort: no GPU kernel yet — omitted so PyTorch handles them silently
        ("add.Tensor",    lambda a, b, alpha=1: _binop_handler(a, b, _BINOP_ADD, alpha)),
        ("mul.Tensor",    lambda a, b: _binop_handler(a, b, _BINOP_MUL)),
        ("sub.Tensor",    lambda a, b, alpha=1: _binop_handler(a, b, _BINOP_SUB, alpha)),
        ("div.Tensor",    lambda a, b: _binop_handler(a, b, _BINOP_DIV)),
    ]

    result: dict = {}
    for path, handler in _raw:
        ns, overload = path.rsplit(".", 1)
        try:
            op = getattr(getattr(aten, ns), overload)
            result[op] = handler
        except AttributeError:
            pass  # overload not in this PyTorch version
    return result
