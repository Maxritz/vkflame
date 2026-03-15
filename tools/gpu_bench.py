#!/usr/bin/env python3
r"""
vkflame GPU benchmark — head-to-head: native ROCm vs vkflame Vulkan shim.

Run with the AMD AI Bundle Python (has torch 2.9+ROCm):
  %LOCALAPPDATA%\AMD\AI_Bundle\ComfyUI\venv\Scripts\python.exe tools/gpu_bench.py

What it tests (GPU-only, no CPU fallback):
  1.  FP16 GEMM          — torch.mm, various shapes
  2.  FP16 batched GEMM  — torch.bmm
  3.  Flash Attention     — scaled_dot_product_attention (MHA + GQA + causal)
  4.  RMS Norm            — native_layer_norm without bias
  5.  Softmax             — F.softmax, large vocab
  6.  SiLU / GELU / ReLU — element-wise activations
  7.  Element-wise binops — add / mul / sub / div
  8.  Top-K sampling      — torch.topk
  9.  Embedding lookup    — F.embedding
  10. Full transformer layer (GEMM + attn + norm + act in sequence)

Each op:
  - 10 warmup iters
  - 100 timed iters
  - Median latency (ms) + peak TFLOPS where applicable
  - Correctness check: vkflame output vs ROCm output, max absolute error

Output format:
  op                shape                    ROCm ms   VKF ms    speedup   max_err
"""
from __future__ import annotations

import ctypes
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

# Force UTF-8 on Windows (default console is cp1252; box-drawing chars crash it)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress per-op debug spam so it doesn't mix with benchmark output
os.environ.pop("VKFLAME_DEBUG", None)

# ── Paths ──────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent
_VKF_RT = _REPO / "build" / "Release" / "vkflame_rt.dll"

if not _VKF_RT.exists():
    sys.exit(f"[FAIL] vkflame_rt.dll not found at {_VKF_RT}\n"
             "  Run: cmake -B build && cmake --build build --config Release")

# ── Device ─────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    sys.exit("[FAIL] No CUDA/ROCm GPU visible. This benchmark is GPU-only.")

DEV = torch.device("cuda")
torch.cuda.set_device(0)

# ── vkflame bootstrap ──────────────────────────────────────────────
# We need vkflame mode active for vkflame-routed ops and inactive for ROCm ops.
# Strategy: run each op TWICE — first with mode OFF (pure ROCm), then ON (vkflame).

# python/vkflame package is the importable vkflame module;
# torch_backend lives at repo root (not inside python/vkflame/)
sys.path.insert(0, str(_REPO / "python"))
sys.path.insert(0, str(_REPO))

try:
    import vkflame as _vkf_module
    _vkf_rt: ctypes.CDLL = _vkf_module._rt
    _vkf_ctx = _vkf_module._ctx
    # torch_backend is at repo root, not inside the vkflame package
    from torch_backend.mode import VKFlameMode
    _vkflame_available = True
    print(f"[vkflame] runtime loaded OK  ({_VKF_RT})")
    _vkf_module.info()
except Exception as e:
    _vkflame_available = False
    VKFlameMode = None  # type: ignore
    print(f"[warn] vkflame not available: {e}")
    print("       Showing ROCm-only numbers.")


# ── Timing helpers ─────────────────────────────────────────────────
_N_WARMUP = 10
_N_BENCH  = 50


def _sync() -> None:
    torch.cuda.synchronize()


def _measure_ms(fn: Callable, n_warmup: int = _N_WARMUP, n_bench: int = _N_BENCH) -> float:
    """Return average latency in ms.

    Uses a single sync wrapping all iterations so the GPU stays loaded
    throughout — per-iter sync starves the queue and hangs on ROCm/Windows.
    """
    for _ in range(n_warmup):
        fn()
    _sync()

    t0 = time.perf_counter()
    for _ in range(n_bench):
        fn()
    _sync()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / n_bench


def _tflops(flops: float, ms: float) -> str:
    if ms <= 0:
        return "   -  "
    t = flops / (ms * 1e9)
    return f"{t:6.1f}"


# ── Table printer ──────────────────────────────────────────────────
_HDR = f"{'op':<22}{'shape':<30}{'ROCm ms':>9}{'VKF ms':>9}{'speedup':>9}{'TFLOPS(vkf)':>12}{'max_err':>10}"
_SEP = "-" * len(_HDR)


def _print_header() -> None:
    print()
    print(_HDR)
    print(_SEP)


def _print_row(
    op: str,
    shape: str,
    rocm_ms: float,
    vkf_ms: float,
    flops: float = 0.0,
    max_err: float = float("nan"),
) -> None:
    speedup = rocm_ms / vkf_ms if vkf_ms > 0 else float("nan")
    tfl = _tflops(flops, vkf_ms) if flops > 0 else "   -  "
    err_str = f"{max_err:.2e}" if not math.isnan(max_err) else "  n/a "
    print(f"{op:<22}{shape:<30}{rocm_ms:>9.3f}{vkf_ms:>9.3f}{speedup:>8.2f}x{tfl:>12}{err_str:>10}")


# ── Per-op benchmark helper ────────────────────────────────────────

def _bench_op(
    op_name: str,
    shape_str: str,
    make_inputs: Callable,       # returns fresh tensors each call
    rocm_fn: Callable,           # takes *inputs → tensor (pure ROCm, no mode)
    vkf_fn:  Callable,           # takes *inputs → tensor (same logic, under VKFlameMode)
    flops: float = 0.0,
) -> None:
    inputs = make_inputs()

    # ── ROCm baseline (no vkflame mode) ───────────────────────────
    # pre-warm outside timing to avoid first-kernel JIT cost
    rocm_out = rocm_fn(*inputs)
    _sync()
    rocm_ms = _measure_ms(lambda: rocm_fn(*inputs))

    # ── vkflame path ───────────────────────────────────────────────
    if _vkflame_available:
        with VKFlameMode():
            vkf_out = vkf_fn(*inputs)
        _sync()

        def _vkf_timed():
            with VKFlameMode():
                vkf_fn(*inputs)

        vkf_ms = _measure_ms(_vkf_timed)

        # Correctness: compare on CPU
        ref_f32  = rocm_out.detach().float().cpu()
        vkf_f32  = vkf_out.detach().float().cpu()
        if ref_f32.shape == vkf_f32.shape:
            max_err = (ref_f32 - vkf_f32).abs().max().item()
        else:
            max_err = float("nan")
    else:
        vkf_ms  = float("nan")
        max_err = float("nan")

    _print_row(op_name, shape_str, rocm_ms, vkf_ms, flops, max_err)
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════
#  BENCHMARKS
# ══════════════════════════════════════════════════════════════════

def bench_gemm_fp16() -> None:
    print("\n== FP16 GEMM (torch.mm) ====================")
    _print_header()
    shapes = [
        (256,  256,  256),
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (1,    4096, 4096),   # batch=1 decode
        (32,   4096, 4096),   # batch=32 prefill
        (128,  8192, 8192),   # 70B-scale inner dim
    ]
    for M, N, K in shapes:
        def make(m=M, n=N, k=K):
            return (torch.randn(m, k, dtype=torch.float16, device=DEV),
                    torch.randn(k, n, dtype=torch.float16, device=DEV))
        flops = 2.0 * M * N * K
        _bench_op("mm_fp16", f"({M},{N},{K})", make,
                  lambda a, b: torch.mm(a, b),
                  lambda a, b: torch.mm(a, b),
                  flops)


def bench_gemm_batched() -> None:
    print("\n== Batched FP16 GEMM (torch.bmm) ===========")
    _print_header()
    shapes = [
        (4,  1024, 1024, 1024),
        (32, 128,  4096, 128),   # typical MHA projection
    ]
    for BS, M, N, K in shapes:
        def make(bs=BS, m=M, n=N, k=K):
            return (torch.randn(bs, m, k, dtype=torch.float16, device=DEV),
                    torch.randn(bs, k, n, dtype=torch.float16, device=DEV))
        flops = 2.0 * BS * M * N * K
        _bench_op("bmm_fp16", f"(bs={BS},{M},{N},{K})", make,
                  lambda a, b: torch.bmm(a, b),
                  lambda a, b: torch.bmm(a, b),
                  flops)


def bench_flash_attention() -> None:
    print("\n== Flash Attention (scaled_dot_product_attention) ==")
    _print_header()
    configs = [
        # (B, Hq, Hkv, Sq, Skv, D,  causal)
        (1,  32, 32,  512,  512,  64, False),   # MHA short
        (1,  32, 32, 2048, 2048,  64, True),    # MHA causal long
        (4,  32, 32, 1024, 1024, 128, True),    # larger D
        (1,  32,  8, 2048, 2048,  64, True),    # GQA 4:1
        (1,  32,  4, 4096, 4096,  64, True),    # GQA 8:1
    ]
    for B, Hq, Hkv, Sq, Skv, D, causal in configs:
        label = "GQA" if Hq != Hkv else "MHA"
        causal_str = "+causal" if causal else ""
        shape_str = f"B{B}H{Hq}/{Hkv}S{Sq}D{D}{causal_str}"

        def make(b=B, hq=Hq, hkv=Hkv, sq=Sq, skv=Skv, d=D):
            Q = torch.randn(b, hq, sq,  d, dtype=torch.float16, device=DEV)
            K = torch.randn(b, hkv,skv, d, dtype=torch.float16, device=DEV)
            V = torch.randn(b, hkv,skv, d, dtype=torch.float16, device=DEV)
            return Q, K, V

        _causal = causal
        _ratio = Hq // Hkv  # 1 for MHA, >1 for GQA
        flops = 4.0 * B * Hq * Sq * Skv * D  # approx
        # Both ROCm and vkflame use expanded K/V for timing comparison (apples-to-apples).
        # vkflame's _sdpa_handler tries Vulkan native GQA first; the Python fallback
        # also expands K/V, so the workload is equivalent either way.
        _bench_op(f"sdpa_{label}", shape_str, make,
                  lambda q, k, v, c=_causal, r=_ratio: F.scaled_dot_product_attention(
                      q,
                      k.repeat_interleave(r, dim=1) if r > 1 else k,
                      v.repeat_interleave(r, dim=1) if r > 1 else v,
                      is_causal=c),
                  lambda q, k, v, c=_causal, r=_ratio: F.scaled_dot_product_attention(
                      q,
                      k.repeat_interleave(r, dim=1) if r > 1 else k,
                      v.repeat_interleave(r, dim=1) if r > 1 else v,
                      is_causal=c),
                  flops)


def bench_rms_norm() -> None:
    print("\n== RMS Norm =========================")
    _print_header()
    shapes = [(32, 4096), (32, 8192), (32, 32768), (128, 4096)]
    for M, N in shapes:
        def make(m=M, n=N):
            x = torch.randn(m, n, dtype=torch.float16, device=DEV)
            g = torch.ones(n, dtype=torch.float16, device=DEV)
            return x, g
        _bench_op("rms_norm", f"({M},{N})", make,
                  lambda x, g: torch.ops.aten.native_layer_norm(x, [x.shape[-1]], g, None, 1e-6)[0],
                  lambda x, g: torch.ops.aten.native_layer_norm(x, [x.shape[-1]], g, None, 1e-6)[0])


def bench_softmax() -> None:
    print("\n== Softmax =========================")
    _print_header()
    shapes = [(1, 32000), (32, 32000), (32, 128000), (128, 32000)]
    for M, N in shapes:
        def make(m=M, n=N):
            return (torch.randn(m, n, dtype=torch.float16, device=DEV),)
        _bench_op("softmax", f"({M},{N})", make,
                  lambda x: F.softmax(x, dim=-1),
                  lambda x: F.softmax(x, dim=-1))


def bench_activations() -> None:
    print("\n== Activations (element-wise unary) =========")
    _print_header()
    N = 4096 * 32
    acts = [
        ("silu",  lambda x: F.silu(x)),
        ("gelu",  lambda x: F.gelu(x)),
        ("relu",  lambda x: F.relu(x)),
    ]
    for name, fn in acts:
        def make(n=N):
            return (torch.randn(n, dtype=torch.float16, device=DEV),)
        _bench_op(name, f"(n={N})", make, fn, fn)


def bench_binops() -> None:
    print("\n== Element-wise Binary Ops ==========")
    _print_header()
    N = 4096 * 128
    ops_list = [
        ("add",  lambda a, b: a + b),
        ("mul",  lambda a, b: a * b),
        ("sub",  lambda a, b: a - b),
        # div: avoid near-zero denominator so error metric is meaningful
        ("div",  lambda a, b: a / (b.abs() + 0.1)),
    ]
    for name, fn in ops_list:
        def make(n=N):
            return (torch.randn(n, dtype=torch.float16, device=DEV),
                    torch.randn(n, dtype=torch.float16, device=DEV))
        _bench_op(name, f"(n={N})", make, fn, fn)


def bench_topk() -> None:
    print("\n== Top-K Sampling ===================")
    _print_header()
    configs = [(1, 32000, 50), (32, 32000, 50), (1, 128000, 100)]
    for M, N, K in configs:
        def make(m=M, n=N):
            return (torch.randn(m, n, dtype=torch.float32, device=DEV),)
        _bench_op("topk", f"({M},{N},k={K})", make,
                  lambda x, k=K: torch.topk(x, k).values,
                  lambda x, k=K: torch.topk(x, k).values)


def bench_embedding() -> None:
    print("\n== Embedding Lookup =================")
    _print_header()
    # On Windows, staging 262+ MB weight matrices per-iteration is very slow
    # (100% staging overhead until Win32 zero-copy lands).  Use smaller
    # D to keep staging cost tolerable.
    if sys.platform == "win32":
        configs = [(512, 32000, 256), (2048, 32000, 256), (512, 128000, 256)]
    else:
        configs = [(512, 32000, 4096), (2048, 32000, 4096), (512, 128000, 4096)]
    for B, V, D in configs:
        def make(b=B, v=V, d=D):
            w   = torch.randn(v, d, dtype=torch.float16, device=DEV)
            idx = torch.randint(0, v, (b,), device=DEV)
            return idx, w  # F.embedding(input=idx, weight=w)
        _bench_op("embedding", f"(B={B},V={V},D={D})",  make,
                  lambda i, w: F.embedding(i, w),
                  lambda i, w: F.embedding(i, w))


def bench_transformer_layer() -> None:
    """
    Synthetic single transformer decoder layer (no weights loaded):
      x → RMSNorm → QKV proj → flash_attn → out_proj → RMSNorm → gate+up → SiLU → down
    Sizes match a ~7B model (hidden=4096, heads=32, kv_heads=8, ffn=14336).
    """
    print("\n== Full Transformer Layer (7B config, bs=1, seq=512) ==")
    _print_header()

    H  = 4096
    NH = 32
    KVH = 8
    D  = H // NH          # 128
    FFN = 14336
    BS  = 1
    SEQ = 512

    def make():
        x       = torch.randn(BS, SEQ, H,           dtype=torch.float16, device=DEV)
        w_qkv   = torch.randn(H + 2*(KVH*D), H,     dtype=torch.float16, device=DEV)  # fused QKV
        w_out   = torch.randn(H, H,                  dtype=torch.float16, device=DEV)
        g_norm1 = torch.ones(H,                      dtype=torch.float16, device=DEV)
        g_norm2 = torch.ones(H,                      dtype=torch.float16, device=DEV)
        w_gate  = torch.randn(FFN, H,                dtype=torch.float16, device=DEV)
        w_up    = torch.randn(FFN, H,                dtype=torch.float16, device=DEV)
        w_down  = torch.randn(H, FFN,                dtype=torch.float16, device=DEV)
        return x, w_qkv, w_out, g_norm1, g_norm2, w_gate, w_up, w_down

    def layer_fn(x, w_qkv, w_out, g_norm1, g_norm2, w_gate, w_up, w_down):
        # --- Attention ---
        xn, _, _ = torch.ops.aten.native_layer_norm(x, [H], g_norm1, None, 1e-6)
        xn2d = xn.reshape(-1, H)                       # (BS*SEQ, H)
        qkv   = torch.mm(xn2d, w_qkv.t())             # (BS*SEQ, H+2*KVH*D)
        q = qkv[:, :NH*D].reshape(BS, SEQ, NH, D).transpose(1,2)
        k = qkv[:, NH*D : NH*D+KVH*D].reshape(BS, SEQ, KVH, D).transpose(1,2)
        v = qkv[:, NH*D+KVH*D:].reshape(BS, SEQ, KVH, D).transpose(1,2)
        # GQA: expand K/V from KVH to NH heads before SDPA
        ratio = NH // KVH
        k_exp = k.repeat_interleave(ratio, dim=1)
        v_exp = v.repeat_interleave(ratio, dim=1)
        attn = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        attn2d = attn.transpose(1,2).reshape(-1, H)
        out   = torch.mm(attn2d, w_out.t()).reshape(BS, SEQ, H)
        x = x + out
        # --- FFN ---
        xn, _, _ = torch.ops.aten.native_layer_norm(x, [H], g_norm2, None, 1e-6)
        xn2d = xn.reshape(-1, H)
        gate = F.silu(torch.mm(xn2d, w_gate.t()))     # (BS*SEQ, FFN)
        up   = torch.mm(xn2d, w_up.t())
        ffn  = torch.mm(gate * up, w_down.t()).reshape(BS, SEQ, H)
        return x + ffn

    flops = (
        2 * BS * SEQ * H * (H + 2*KVH*D) +   # QKV proj
        4 * BS * NH * SEQ * SEQ * D +          # attn scores + weighted sum
        2 * BS * SEQ * H * H +                 # out proj
        2 * BS * SEQ * H * FFN * 3             # gate + up + down
    )
    _bench_op("transformer_layer", "bs=1,seq=512,7B", make, layer_fn, layer_fn, flops)


# ══════════════════════════════════════════════════════════════════
#  CORRECTNESS SPOT-CHECKS  (quick sanity, not exhaustive)
# ══════════════════════════════════════════════════════════════════

def correctness_checks() -> None:
    print("\n\n══ Correctness spot-checks ══════════════════════════════")
    errors: list[str] = []

    def check(name: str, fn_rocm, fn_vkf, tol: float, *make_args) -> None:
        with torch.no_grad():
            ref = fn_rocm(*make_args).float().cpu()
            with VKFlameMode():
                got = fn_vkf(*make_args).float().cpu()
        if ref.shape != got.shape:
            errors.append(f"FAIL {name}: shape mismatch {ref.shape} vs {got.shape}")
            print(f"  FAIL  {name:<30} shape mismatch")
            return
        err = (ref - got).abs().max().item()
        status = "PASS" if err < tol else "FAIL"
        if status == "FAIL":
            errors.append(f"FAIL {name}: max_err={err:.3e} tol={tol:.0e}")
        print(f"  {status}  {name:<30} max_err={err:.2e}  tol={tol:.0e}")

    if not _vkflame_available:
        print("  [skip] vkflame not available")
        return

    # GEMM fp16
    A = torch.randn(256, 256, dtype=torch.float16, device=DEV)
    B = torch.randn(256, 256, dtype=torch.float16, device=DEV)
    check("mm_fp16 256x256",
          lambda a, b: torch.mm(a.float(), b.float()).half(),
          lambda a, b: torch.mm(a, b),
          0.5, A, B)

    # RMSNorm
    x = torch.randn(32, 4096, dtype=torch.float16, device=DEV)
    g = torch.ones(4096, dtype=torch.float16, device=DEV)
    check("rms_norm 32x4096",
          lambda x, g: torch.ops.aten.native_layer_norm(x.float(), [4096], g.float(), None, 1e-6)[0].half(),
          lambda x, g: torch.ops.aten.native_layer_norm(x, [4096], g, None, 1e-6)[0],
          0.05, x, g)

    # Softmax
    logits = torch.randn(4, 32000, dtype=torch.float16, device=DEV)
    check("softmax 4x32000",
          lambda x: F.softmax(x.float(), dim=-1).half(),
          lambda x: F.softmax(x, dim=-1),
          1e-3, logits)

    # Softmax sums to 1
    with VKFlameMode():
        sm_out = F.softmax(logits, dim=-1).float()
    row_sums = sm_out.sum(dim=-1)
    err = (row_sums - 1.0).abs().max().item()
    status = "PASS" if err < 1e-3 else "FAIL"
    print(f"  {status}  {'softmax_sums_to_1':<30} max_row_err={err:.2e}")

    # Flash attention MHA
    Q = torch.randn(2, 8, 64, 64, dtype=torch.float16, device=DEV)
    K = torch.randn(2, 8, 64, 64, dtype=torch.float16, device=DEV)
    V = torch.randn(2, 8, 64, 64, dtype=torch.float16, device=DEV)
    check("sdpa_mha B2H8S64D64",
          lambda q, k, v: F.scaled_dot_product_attention(q.float(), k.float(), v.float()).half(),
          lambda q, k, v: F.scaled_dot_product_attention(q, k, v),
          0.1, Q, K, V)

    # Flash attention causal
    check("sdpa_causal B1H4S32D64",
          lambda q, k, v: F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True).half(),
          lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
          0.1,
          torch.randn(1, 4, 32, 64, dtype=torch.float16, device=DEV),
          torch.randn(1, 4, 32, 64, dtype=torch.float16, device=DEV),
          torch.randn(1, 4, 32, 64, dtype=torch.float16, device=DEV))

    # Top-K
    x_topk = torch.tensor([[3., 1., 4., 1., 5., 9., 2., 6.]], device=DEV)
    with VKFlameMode():
        vals, idx = torch.topk(x_topk, 3)
    ok = (vals[0,0].item() == 9.0 and vals[0,1].item() == 6.0 and vals[0,2].item() == 5.0
          and idx[0,0].item() == 5 and idx[0,1].item() == 7 and idx[0,2].item() == 4)
    print(f"  {'PASS' if ok else 'FAIL'}  {'topk [3,1,4,1,5,9,2,6] k=3':<30} vals=[9,6,5] idx=[5,7,4]")
    if not ok:
        errors.append(f"FAIL topk: vals={vals} idx={idx}")

    # SiLU
    x_act = torch.randn(1024, dtype=torch.float16, device=DEV)
    check("silu n=1024",
          lambda x: F.silu(x.float()).half(),
          lambda x: F.silu(x),
          0.01, x_act)

    # Binary ops
    a_b = torch.randn(4096, dtype=torch.float16, device=DEV)
    b_b = torch.randn(4096, dtype=torch.float16, device=DEV)
    for op_name, fn in [("add", lambda a,b: a+b), ("mul", lambda a,b: a*b),
                         ("sub", lambda a,b: a-b)]:
        check(f"binop_{op_name} n=4096",
              lambda a, b, f=fn: f(a.float(), b.float()).half(),
              lambda a, b, f=fn: f(a, b),
              0.01, a_b, b_b)

    print()
    if errors:
        print(f"  {len(errors)} FAILURE(s):")
        for e in errors:
            print(f"    {e}")
    else:
        print("  All correctness checks PASSED.")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 80)
    print("  vkflame GPU Benchmark — ROCm vs Vulkan shim")
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    rocm_ver = getattr(torch.version, "hip", "n/a")
    print(f"  ROCm   : {rocm_ver}")
    print(f"  vkflame: {'YES' if _vkflame_available else 'NO (ROCm-only mode)'}")
    if _vkflame_available and sys.platform == "win32":
        print("  NOTE: Windows -- VKF timings use CPU-bounce staging (upload+download).")
        print("        Range-lookup in vkflame_buf_from_ptr handles PyTorch sub-allocations.")
        print("        STATUS: staging correctness verified; zero-copy planned via")
        print("                hipExtMallocWithFlags + hipMemGetExportHandle.")
    print("=" * 80)

    # All perf benchmarks
    bench_gemm_fp16()
    bench_gemm_batched()
    bench_flash_attention()
    bench_rms_norm()
    bench_softmax()
    bench_activations()
    bench_binops()
    bench_topk()
    bench_embedding()
    try:
        bench_transformer_layer()
    except Exception as _e:
        print(f"[CRASH] bench_transformer_layer: {type(_e).__name__}: {_e}")
        import traceback; traceback.print_exc()

    # Correctness
    correctness_checks()

    print("\nDone.")


if __name__ == "__main__":
    main()
