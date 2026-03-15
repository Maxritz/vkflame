# vkflame shim — next-steps plan

Last commit in: `f338cf2`  
Current branch: `main`  
GPU: AMD RX 9070 XT (gfx1201 RDNA4), WDDM Windows

---

## What landed since f338cf2 (uncommitted)

| File | Change |
|------|--------|
| `runtime/buffer.cpp` | CB-batching infra: `vkflame_batch_begin/end`, staging-offset tracking, deferred-pool list |
| `runtime/buffer.h` | `vkflame_batch_begin/end` exported; batch helper prototypes |
| `runtime/dispatch.cpp` | Batch-aware `do_dispatch`: reuses batch CB or allocs fresh one; TRANSFER→COMPUTE barrier injection |
| `runtime/pipeline.cpp` | Wave32 enforcement for kernels 7 & 8 via `VK_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO` |
| `runtime/device.cpp` | Use `minSubgroupSize` from subgroup-size-control props → correctly reports 32 on RDNA4 |
| `kernels/linear_fp16_fp32out.glsl` | Output buffer type fixed: was `float`, now `float16_t`; accumulator stays `float` internally |
| `torch_backend/ops.py` | `_staged()` wraps in `batch_begin/batch_end` when runtime exports those symbols |
| `tools/gpu_bench.py` | Per-iter timing + 3 s deadline cap + median; `_N_WARMUP=5`, `_N_BENCH=20` |

---

## Active defects

### P0 — TDR / device lost (blocks all benchmarks)

`VKF_CHECK` fails at `buffer.cpp:129` and `buffer.cpp:223` interleaved into
bench stdout. The GPU hit TDR before completing the mm/rms_norm section.

Likely causes (in order):
1. **Batch CB staging-offset race**: `g_batch_stg_off` is advanced per h2d call but
   never validated against `g_staging.capacity`. If a batch overflows the 256 MB
   arena, the copy descriptor points outside mapped memory → device lost.
2. **Missing COMPUTE→TRANSFER barrier before d2h in batch mode**: the batch
   `vkflame_memcpy_d2h` records a copy but `g_batch_need_xfer_barrier` is only
   set in `vkf_batch_pre_compute_barrier()`; if no compute preceded, the flag
   is never cleared and the barrier is never inserted.
3. **Descriptor pool destroyed before batch fence wait**: if `vkf_batch_defer_pool`
   list isn't cleared after fence wait in `vkflame_batch_end`, pools accumulate
   across calls.

**Fix path:**
```cpp
// In vkflame_batch_begin():
g_batch_stg_off = 0;  // reset per-batch; must not exceed g_staging.capacity
// In h2d (batch mode), before recording copy:
assert(g_batch_stg_off + size <= g_staging.capacity);
```
Add explicit COMPUTE→TRANSFER barrier in `vkflame_memcpy_d2h` when `g_batch_need_xfer_barrier`.

Also add a hard fallback: if `g_batch_stg_off + size > capacity`, flush current
batch early (submit + wait), reset offset, open a new batch CB.

### P1 — Precision errors: mm fp16

`max_err` at large K:
- `(256,256,256)` → 0.031
- `(1024,1024,1024)` → 0.063
- `(4096,4096,4096)` → 0.25
- `(128,8192,8192)` → 0.25

Root cause: `linear_fp16.glsl` accumulates in `float16_t` — error accumulates
as `O(K * ε²)`. At K=8192 with ε=5×10⁻⁴ this gives max drift ~0.21.

**Fix** (`kernels/linear_fp16.glsl` — one-line change in inner loop):
```glsl
// Replace:
float16_t acc = float16_t(0.0);
for (uint k = 0u; k < pc.K; k++)
    acc += a[row * pc.K + k] * b[k * pc.N + col];
y[...] = acc;

// With:
float acc = 0.0;                             // fp32 accumulator
for (uint k = 0u; k < pc.K; k++)
    acc += float(a[row * pc.K + k]) * float(b[k * pc.N + col]);
y[...] = float16_t(acc);                     // write fp16 output
```
No new kernel variant, no routing change, no ops.py change.
Target: max_err ≤ 0.01 at all shapes.

### P2 — Precision errors: rms_norm fp16

`max_err` up to 0.051 at `(128, 4096)`.

Root cause: `rms_norm.glsl` accumulates `sum_sq` in `float16_t` — saturates at
large N because individual squares can overflow fp16 range before the reduction.

**Fix** (`kernels/rms_norm.glsl`):
```glsl
// Change sum-of-squares accumulation to fp32 only:
float sum_sq = 0.0;                          // was: float16_t
for (uint i = threadId; i < pc.N; i += 256) {
    float v = float(x[row * pc.N + i]);
    sum_sq += v * v;
}
// subgroupAdd on float is fine; barrier / rrms calculation unchanged.
// Output write stays float16_t(result * rrms * float16_t(g[col])).
```
Target: max_err ≤ 0.005.

---

## Pending small tasks

| Task | File | Detail |
|------|------|--------|
| UTF-8 stdout | `tools/gpu_bench.py` | add `import sys; sys.stdout.reconfigure(encoding='utf-8')` at top |
| rms_norm_f32 / fp32out variants | `kernels/` | keep (used by `vkflame_dispatch_rms_norm_f32`) but ensure never accidentally routed for normal fp16 inference |
| Verify subgroup=32 fix prints correctly | `runtime/device.cpp` | `[vkflame] … subgroup:32` vs old `subgroup:64` |
| Delete `bench_out.txt` etc. from repo | `.gitignore` | add `bench_out.txt bench_stderr.txt bench_stdout.txt tools/bench_err.txt` |

---

## Performance target

Transformer layer (32 ops, fp16, seq=1): `< 20 ms` end-to-end.

Current: ~780 ms (staging + 3 × vkQueueSubmit overhead per op).  
Expected after CB batching stabilised: ~50-80 ms (1 submit per op, WDDM overhead ~3ms/submit × 32 = 96ms minimum).  
After zero-copy (future): ~5-10 ms.

CB batching collapses 3 submits to 1 per `_staged()` call. Once P0 TDR is fixed
and batch mode is stable, run:
```powershell
$py tools/gpu_bench.py 2>&1 | Select-String "transformer"
```

---

## Build & test commands

```powershell
# Build
cd C:\TR\vkflame
cmake --build build --config Release 2>&1 | Select-Object -Last 15

# Recompile shaders
cmake --build build --config Release --target spirv_kernels 2>&1 | Select-Object -Last 10

# Quick correctness
$env:PYTHONPATH="python"
$py = "$env:LOCALAPPDATA\AMD\AI_Bundle\ComfyUI\venv\Scripts\python.exe"
& $py -c "import vkflame; vkflame.info()"

# Bench (just mm + rms_norm)
& $py tools/gpu_bench.py 2>&1 | Select-String "mm_fp16|rms_norm" | Select-Object -First 20
```
