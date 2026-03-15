# vkflame

**Run ROCm/HIP AI workloads on any GPU — AMD GCN/RDNA/CDNA, NVIDIA, Intel — via Vulkan.**

vkflame is a drop-in replacement for the AMD ROCm/HIP runtime.
It ships as DLLs (Windows) / shared objects (Linux) that stand in for
`amdhip64`, `hipblas`, and `hipblaslt`.

Any application compiled against ROCm/HIP — Ollama, llama.cpp, ggml, PyTorch-ROCm,
any HIP binary — will run through vkflame's Vulkan 1.3 compute backend with **no
recompilation, no code changes, no ROCm installation required on the target machine**.

The math is ours. Every kernel is a hand-written GLSL compute shader compiled to
SPIR-V. There is no dependency on AMD's proprietary libraries at runtime.

**Confirmed working:** DeepSeek R1 Distill Qwen 14B fully offloaded (49/49 layers)
to AMD Radeon RX 9070 XT (gfx1201, RDNA4) via Ollama on Windows. Flash Attention
auto-enabled. 8 GiB model + 768 MiB KV-cache on GPU.

---

## How it works

```
App (Ollama / llama.cpp / PyTorch / any HIP binary)
  │
  ├─ amdhip64_6.dll   ← hipMalloc, hipMemcpy, hipLaunchKernel, device queries
  ├─ hipblas.dll      ← hipblasSgemm, hipblasHgemm, batched variants
  ├─ hipblaslt.dll    ← hipblasLtMatmul with epilogue fusions
  │
  └─ vkflame_rt.dll   ← Vulkan 1.3 backend (no ROCm required)
       ├─ device.cpp      VkInstance, VkDevice, queue families, feature detection
       ├─ buffer.cpp      VkBuffer alloc, staging copies, address→buffer map
       ├─ pipeline.cpp    SPIR-V loading, pipeline cache, descriptor sets
       └─ dispatch.cpp    Push-constant fill, vkCmdDispatch, submit+wait
```

All GPU memory is `VkBuffer`. Device addresses returned by `hipMalloc` are
`VkDeviceAddress` values. `hipLaunchKernel` looks up the kernel name in a
dispatch table and calls the matching Vulkan compute shader.

The math is ours. Every kernel is hand-written GLSL compiled to SPIR-V.
No AMD proprietary libraries are used or required at runtime.
If the target machine has any GPU with Vulkan 1.3, it runs.

---

## GLSL Compute Kernels (25 shaders)

| Kernel           | File                                                   | Notes                                            |
| ---------------- | ------------------------------------------------------ | ------------------------------------------------ |
| FP16 GEMM        | `linear_fp16.glsl`                                     | 16×16 workgroup tiling                           |
| INT8 GEMM        | `linear_int8.glsl`                                     | `GL_EXT_integer_dot_product`, per-channel scale  |
| Coop-matrix GEMM | `linear_coop.glsl`                                     | `GL_KHR_cooperative_matrix` (RDNA4+)             |
| Flash Attention  | `flash_attention.glsl`                                 | MHA, GQA, causal; online softmax across KV tiles |
| RMS Norm (FP16)  | `rms_norm.glsl`                                        | subgroup reduce                                  |
| RMS Norm (FP32)  | `rms_norm_f32.glsl`                                    | ggml f32 path                                    |
| Online Softmax   | `softmax_online.glsl`                                  | numerically stable, subgroup merge               |
| Dequant Q4_0     | `dequant_q4_0.glsl`                                    | ggml block format                                |
| Dequant Q4_1     | `dequant_q4_1.glsl`                                    |                                                  |
| Dequant Q4_K     | `dequant_q4_k.glsl`                                    |                                                  |
| Dequant Q5_0     | `dequant_q5_0.glsl`                                    |                                                  |
| Dequant Q5_K     | `dequant_q5_k.glsl`                                    |                                                  |
| Dequant Q6_K     | `dequant_q6_k.glsl`                                    |                                                  |
| Dequant Q8_0     | `dequant_q8_0.glsl`                                    |                                                  |
| Dequant Q8_1     | `dequant_q8_1.glsl`                                    | ggml activation quantisation (36-byte blocks)    |
| Elementwise FP32 | `elementwise_f32.glsl`                                 | silu, gelu (erf), relu, tanh                     |
| Binary Op FP32   | `binop_f32.glsl`                                       | add, mul, sub, div                               |
| RoPE NeoX        | `rope_neox.glsl`                                       | rotary position embedding                        |
| Scale FP32       | `scale_f32.glsl`                                       | in-place scale                                   |
| Top-K            | `topk.glsl`                                            | selection sort, largest/smallest                 |
| Embedding        | `embedding.glsl`                                       | vocab lookup, bounds-checked                     |
| KV-Cache Update  | `kvcache_update.glsl`                                  | in-place seq_pos write                           |
| Winograd F(2,3)  | `winograd_f23.glsl` + `winograd_filter_transform.glsl` | 3×3 conv                                         |

---

## Requirements

- GPU with **Vulkan 1.3** support
- Windows 10+ or Linux
- CMake >= 3.20
- Vulkan SDK (`glslc`, `spirv-val`)
- Python 3.10+ (optional — for PyTorch backend and tools)

---

## GPU Compatibility

Any GPU that supports Vulkan 1.3 works. vkflame does not call into any AMD/NVIDIA driver-specific compute stack.

| GPU family                | Examples               | Status                               |
| ------------------------- | ---------------------- | ------------------------------------ |
| AMD RDNA4 (gfx1201)       | RX 9070, RX 9070 XT    | ✅ Native — no override needed       |
| AMD RDNA3 (gfx1100)       | RX 7900 XTX, RX 7600   | ✅ `HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| AMD RDNA2 (gfx1030)       | RX 6900 XT, RX 6600 XT | ✅ `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| AMD RDNA1 (gfx1010)       | RX 5700 XT, RX 5500 XT | ✅ `HSA_OVERRIDE_GFX_VERSION=10.1.0` |
| AMD GCN4/Polaris (gfx803) | RX 480, RX 580         | ✅ `HSA_OVERRIDE_GFX_VERSION=9.0.0`  |
| AMD CDNA1/2 (gfx908/90a)  | Instinct MI100, MI250X | ✅ `HSA_OVERRIDE_GFX_VERSION=9.0.8`  |
| NVIDIA Turing+            | RTX 2080, 3090, 4090   | ✅ Vulkan 1.3 — no env var needed    |
| NVIDIA Ampere             | RTX 3060, A100         | ✅ Vulkan 1.3 — no env var needed    |
| Intel Xe / Arc            | Arc A770, A580         | ✅ Vulkan 1.3 — no env var needed    |
| Intel non-Arc (Gen9+)     | UHD 630, Iris Xe       | ⚠️ Vulkan 1.2 only — partially works |

**One-line rule:** if the program was compiled against the ROCm/HIP SDK and your GPU
has Vulkan 1.3, vkflame makes it run regardless of vendor.

On Windows, set env vars in PowerShell before launching:

```powershell
# Example: RDNA3
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
ollama serve
```

Or permanently in System → Advanced → Environment Variables.

---

## Quickstart: AMD bundle Ollama on Windows (prebuilt DLLs)

The AMD bundle Ollama ships ROCm DLLs that only run on RDNA4. vkflame replaces
them with Vulkan shims so the ROCm path runs on any Vulkan 1.3 GPU.

**No ROCm installation required.** Download the release zip and drop the DLLs in.

1. Download `vkflame-windows-x64.zip` from [Releases](../../releases)
2. Find your Ollama ROCm directory:
   - AMD bundle installer: `%LOCALAPPDATA%\AMD\AI_Bundle\Ollama\lib\ollama\rocm\`
   - Standard installer: `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\rocm\`
3. Copy all four DLLs there (overwrite the originals — **back them up first**):
   ```
   amdhip64_6.dll
   hipblas.dll
   hipblaslt.dll
   vkflame_rt.dll
   ```
4. For non-RDNA4 GPUs, set the arch env var (see GPU Compatibility table above)
5. Start Ollama:
   ```
   ollama serve
   ```
   Success: log shows `library=ROCm` (not `library=cpu`)
6. Run a model:
   ```
   ollama run deepseek-r1:14b
   ```

---

## Build from source

### Windows (Visual Studio 2022 or Ninja + MSVC)

```cmd
git clone https://github.com/Maxritz/vkflame
cd vkflame
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target spirv_kernels
cmake --build build
```

Output DLLs in `build\Release\`:

- `vkflame_rt.dll`
- `amdhip64_6.dll`
- `hipblas.dll`
- `hipblaslt.dll`

### Linux

```bash
git clone https://github.com/Maxritz/vkflame
cd vkflame
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc)
```

---

## Use with any ROCm/HIP binary (Windows)

Drop the four DLLs in the same directory as the executable (or anywhere earlier on
`PATH` than the real ROCm installation). Windows DLL search order does the rest.

```cmd
copy build\Release\*.dll C:\path\to\app\
C:\path\to\app\your_hip_program.exe
```

---

## Use with any ROCm/HIP binary (Linux — LD_PRELOAD)

```bash
python -m vkflame.install          # writes LD_PRELOAD to ~/.config/vkflame/env.sh
source ~/.config/vkflame/env.sh    # or open a new terminal
./any_hip_binary
```

---

## Quantise a model

```bash
python -m vkflame.tools.quantise /path/to/hf-model /path/to/output-int8
```

Converts all `torch.nn.Linear` layers to INT8 (symmetric per-output-channel).
Run once; reuse the output for all inference sessions.

---

## Use with PyTorch

```python
import vkflame
import torch

vkflame.install(permanent=True)   # intercept all aten ops globally

A = torch.randn(4096, 4096, dtype=torch.float16).cuda()
B = torch.randn(4096, 4096, dtype=torch.float16).cuda()
C = torch.mm(A, B)   # dispatched to vkflame Vulkan kernel
```

Or as a context manager:

```python
with vkflame.install():
    C = torch.mm(A, B)
```

---

## What works

- **Any Vulkan 1.3 GPU**: AMD (GCN, RDNA1/2/3/4, CDNA), NVIDIA (Turing+), Intel Xe
- Full ROCm/HIP ABI: any ROCm-compiled binary loads vkflame DLLs instead of real ROCm
- No ROCm installation required on the target machine — only the Vulkan runtime
- GPU detection: `hipGetDeviceProperties` returns correct arch, VRAM, subgroup size
- Memory: `hipMalloc` / `hipFree` / `hipMemcpy` (H2D, D2H, D2D) / `hipMemset`
- Synchronisation: streams, events, `hipDeviceSynchronize`
- FP16 / FP32 GEMM (`hipblasSgemm`, `hipblasHgemm`, batched + strided)
- `hipblasLtMatmul` with RELU / GELU / SILU epilogue fusions
- All ggml quantised formats: Q4_0, Q4_1, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, Q8_1
- RoPE (NeoX), RMS Norm, Scale, Elementwise, BinOp
- Flash Attention (MHA, GQA, causal mask)
- Top-K, Embedding, KV-cache update
- PyTorch aten op dispatch (mm, addmm, linear, sdpa, layer_norm, softmax, ...)
- **Confirmed: AMD bundle Ollama full model offload, RX 9070 XT (gfx1201), 49/49 layers**

## In progress / planned

These features are on the roadmap. PRs welcome.

| Feature | Notes |
| --- | --- |
| FP8 weights | Needs `VK_KHR_shader_float8` — not yet in Mesa/AMDVLK; shaders written, gated on driver |
| BFloat16 kernels | Needed for Llama3/Mistral native weights; GLSL compiler support landed in 2025 |
| Async compute | All dispatch is currently synchronous (submit+wait); needs timeline semaphores |
| `hipblaslt` workspace hints | Some callers pass workspace size > 0; vkflame ignores it today |
| IQ2 / IQ3 / IQ4 ggml formats | Activations skipped with log warning; full dequant shaders needed |
| Multi-GPU | Single device only today; VkPhysicalDevice enumeration exists, selection does not |

## Out of scope

- **Training** — vkflame is inference-only (text gen, image gen, embeddings). No autograd, no backward pass.
- **Windows LD_PRELOAD** — DLL drop-in via PATH order is the Windows equivalent; no LD_PRELOAD needed.

---

## Running tests

```bash
# Correctness tests (GPU required)
pytest tests/test_correctness.py -v

# With a quantised model for end-to-end test
VKFLAME_TEST_MODEL=/path/to/quantised pytest tests/test_correctness.py -v

# Benchmark
python tools/benchmark.py

# Verbose dispatch logging
VKFLAME_DEBUG=1 python your_script.py
```

On Windows, run the C++ test suite directly:

```powershell
.\build\Release\vkflame_tests.exe
```

---

## Architecture

```
vkflame/
├── kernels/          GLSL compute shaders -> compiled to SPIR-V by glslc
├── runtime/
│   ├── device.cpp    Vulkan instance/device/queue init, feature detection
│   ├── buffer.cpp    VkBuffer alloc, host<->device copies, address map
│   ├── pipeline.cpp  SPIR-V load, pipeline cache, descriptor sets
│   └── dispatch.cpp  Per-op push-constant fill, vkCmdDispatch, submit
├── shim/
│   ├── hip_runtime_shim.cpp   amdhip64 -- full HIP runtime surface
│   ├── hipblas_shim.cpp       hipBLAS GEMM surface
│   ├── hipblaslt_shim.cpp     hipBLASLt matmul + epilogue surface
│   └── rocblas_shim.cpp       rocBLAS stub
├── torch_backend/
│   ├── mode.py       TorchDispatchMode -- intercepts aten ops
│   └── ops.py        Handler implementations (ctypes -> dispatch.cpp)
├── python/vkflame/
│   ├── __init__.py   Load libvkflame_rt, call vkflame_init
│   └── install.py    LD_PRELOAD installer (Linux)
└── tools/
    ├── embed_spirv.py   SPIR-V -> C++ header for link-time embedding
    ├── quantise.py      INT8 per-channel quantisation of HF models
    └── benchmark.py     Latency/TFLOPS table for all op types
```

---

## License

MIT
