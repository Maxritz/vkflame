#!/usr/bin/env python3
"""
vkflame benchmark — measures latency and throughput for critical op types.

Works on Windows 11 (CPU tensors, Vulkan-staged) and Linux/ROCm (CUDA tensors).

Usage: python tools/benchmark.py
"""
import os
import math
import statistics
import time
from typing import Callable

import torch

# Ensure vkflame is on the path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import vkflame
vkflame.install(permanent=True)

# Use "cuda" on Linux+ROCm, "cpu" on Windows (vkflame stages uploads)
_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _G(t: torch.Tensor) -> torch.Tensor:
    return t.to(_DEV)


def _sync() -> None:
    """Ensure GPU work is complete before recording time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _median_ms(fn: Callable, n_warmup: int = 5, n_bench: int = 50) -> float:
    """Return median latency in milliseconds."""
    for _ in range(n_warmup):
        fn()
    _sync()

    times: list[float] = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    return statistics.median(times)


def _tflops(flops: float, ms: float) -> str:
    t = flops / (ms * 1e9)  # ms → seconds * 1e9 = TFLOPS denominator
    return f"{t:.1f}"


def _row(op: str, shape: str, ms: float, tflops_str: str = "-") -> None:
    print(f"{op:<16}{shape:<28}{ms:<10.3f}{tflops_str}")


def bench_mm() -> None:
    shapes = [
        (1,    4096, 4096),
        (32,   4096, 4096),
        (4096, 4096, 4096),
    ]
    for M, N, K in shapes:
        A = _G(torch.randn(M, K, dtype=torch.float16))
        B = _G(torch.randn(K, N, dtype=torch.float16))
        ms = _median_ms(lambda: torch.mm(A, B))
        flops = 2.0 * M * N * K
        _row("mm", f"({M}, {N}, {K})", ms, _tflops(flops, ms))


def bench_flash_attention() -> None:
    configs = [
        (1, 32, 512,  64),
        (1, 32, 2048, 64),
    ]
    for B, H, S, D in configs:
        Q = _G(torch.randn(B, H, S, D, dtype=torch.float16))
        K = _G(torch.randn(B, H, S, D, dtype=torch.float16))
        V = _G(torch.randn(B, H, S, D, dtype=torch.float16))
        ms = _median_ms(
            lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V))
        flops = 4.0 * B * H * S * S * D
        _row("flash_attn", f"({B}, {H}, {S}, {D})", ms, _tflops(flops, ms))


def bench_rms_norm() -> None:
    for M, N in [(32, 4096), (32, 32768)]:
        x     = _G(torch.randn(M, N, dtype=torch.float16))
        gamma = _G(torch.ones(N, dtype=torch.float16))
        ms = _median_ms(
            lambda: torch.ops.aten.native_layer_norm(x, [N], gamma, None, 1e-6))
        _row("rms_norm", f"({M}, {N})", ms)


def bench_softmax() -> None:
    for M, N in [(32, 32000), (32, 128000)]:
        x = _G(torch.randn(M, N, dtype=torch.float16))
        ms = _median_ms(lambda: torch.softmax(x, dim=-1))
        _row("softmax", f"({M}, {N})", ms)


def bench_topk() -> None:
    M, N, K = 32, 32000, 50
    x = _G(torch.randn(M, N, dtype=torch.float32))
    ms = _median_ms(lambda: torch.topk(x, K))
    _row("topk", f"({M}, {N}, k={K})", ms)


def bench_embedding() -> None:
    V, D, B = 32000, 4096, 512
    weight  = _G(torch.randn(V, D, dtype=torch.float16))
    indices = _G(torch.randint(0, V, (B,)))
    ms = _median_ms(lambda: torch.nn.functional.embedding(weight, indices))
    _row("embedding", f"({B}, {V}, {D})", ms)


def main() -> None:
    print(f"[vkflame] device: {_DEV}")
    print(f"{'op':<16}{'shape':<28}{'ms':<10}{'TFLOPS'}")
    print("-" * 64)

    bench_mm()
    bench_flash_attention()
    bench_rms_norm()
    bench_softmax()
    bench_topk()
    bench_embedding()


if __name__ == "__main__":
    main()
