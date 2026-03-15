# vkflame

**Vulkan-native HIP/ROCm shim for AI inference.**

vkflame replaces the AMD ROCm/HIP stack with a pure Vulkan 1.3 compute backend.
It ships as drop-in DLLs (Windows) / shared objects (Linux) that stand in for
`amdhip64`, `hipblas`, and `hipblaslt`, so any app that uses those libraries —
Ollama, llama.cpp, ggml, PyTorch-ROCm — runs through Vulkan with no recompilation.

**Confirmed working:** DeepSeek R1 Distill Qwen 14B fully offloaded (49/49 layers)
to AMD Radeon RX 9070 XT (gfx1201, RDNA4) via Ollama on Windows. Flash Attention
auto-enabled. 8 GiB model + 768 MiB KV-cache on GPU.

---

## How it works

```
App (Ollama / llama.cpp / PyTorch)
  │
  ├─ amdhip64_6.dll   ← hipMalloc, hipMemcpy, hipLaunchKernel, device queries
  ├─ hipblas.dll      ← hipblasSgemm, hipblasHgemm, batched variants
  ├─ hipblaslt.dll    ← hipblasLtMatmul with epilogue fusions
  │
  └─ vkflame_rt.dll   ← Vulkan backend
       ├─ device.cpp      VkInstance, VkDevice, queue families, feature detection
       ├─ buffer.cpp      VkBuffer alloc, staging copies, address→buffer map
       ├─ pipeline.cpp    SPIR-V loading, pipeline cache, descriptor sets
       └─ dispatch.cpp    Push-constant fill, vkCmdDispatch, submit+wait
```

All GPU memory is `VkBuffer`. Device addresses returned by `hipMalloc` are
`VkDeviceAddress` values. `hipLaunchKernel` looks up the kernel name in a
dispatch table and calls the matching Vulkan compute shader.

---

## GLSL Compute Kernels (24 shaders)

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

- GPU with **Vulkan 1.3** support (AMD RDNA2+, NVIDIA Turing+, Intel Xe)
- Windows 10+ or Linux
- CMake >= 3.20
- Vulkan SDK (`glslc`, `spirv-val`)
- Python 3.10+ (optional — for PyTorch backend and tools)

---

## Quickstart: Ollama on Windows (prebuilt DLLs)

**No build required.** Download the release zip and drop the DLLs in.

1. Download `vkflame-windows-x64.zip` from [Releases](../../releases)
2. Find your Ollama ROCm directory:
   - Standard installer: `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\rocm\`
   - AMD bundle installer: `%LOCALAPPDATA%\AMD\AI_Bundle\Ollama\lib\ollama\rocm\`
3. Copy all four DLLs there (overwrite the originals — **back them up first**):
   ```
   amdhip64_6.dll
   hipblas.dll
   hipblaslt.dll
   vkflame_rt.dll
   ```
4. Start Ollama:
   ```
   ollama serve
   ```
   Look for this line in the log:
   ```
   inference compute id=0 library=ROCm compute=gfx1201 name=ROCm0
   description="AMD Radeon RX 9070 XT" total="15.9 GiB" available="14.5 GiB"
   ```
5. Run a model:
   ```
   ollama run deepseek-r1:14b
   ```

See [INSTALL_OLLAMA_WINDOWS.md](INSTALL_OLLAMA_WINDOWS.md) for detailed steps
and troubleshooting.

### GPU compatibility (non-RDNA4)

vkflame defaults to `gfx1201` (RDNA4). For other GPUs set one of these env vars
**before** starting Ollama:

| GPU family                                    | Env var to set                    |
| --------------------------------------------- | --------------------------------- |
| RX 9070 / 9070 XT (RDNA4, gfx1201)            | Nothing — default                 |
| RX 7900 / 7800 / 7700 / 7600 (RDNA3, gfx1100) | `HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| RX 6900 / 6800 / 6700 / 6600 (RDNA2, gfx1030) | `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| RX 6500 / 6400 (RDNA2, gfx1013)               | `HSA_OVERRIDE_GFX_VERSION=10.1.3` |
| Any GPU (explicit override)                   | `VKFLAME_GFX_ARCH=gfxXXXX`        |

On Windows set env vars in PowerShell before launching Ollama:

```powershell
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
ollama serve
```

Or permanently in System → Advanced → Environment Variables.

---

## Build from source

### Windows (Visual Studio 2022 Developer Command Prompt)

```cmd
git clone https://github.com/Maxritz/vkflame
cd vkflame
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target spirv_kernels
cmake --build build
```

Output DLLs are in `build\Release\`:

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

## Use with llama.cpp / any HIP binary (Linux)

```bash
python -m vkflame.install          # writes LD_PRELOAD to ~/.config/vkflame/env.sh
source ~/.config/vkflame/env.sh    # or open a new terminal
./llama-cli -m model.gguf -p "Hello"
```

---

## What works

- GPU detection: `hipGetDeviceProperties` returns correct arch, VRAM, subgroup size
- Memory: `hipMalloc` / `hipFree` / `hipMemcpy` (H2D, D2H, D2D) / `hipMemset`
- Synchronisation: streams, events, `hipDeviceSynchronize`
- FP16 / FP32 GEMM (`hipblasSgemm`, `hipblasHgemm`, batched + strided)
- `hipblasLtMatmul` with RELU / GELU / SILU epilogue fusions
- All ggml quantised formats: Q4_0, Q4_1, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0
- RoPE (NeoX), RMS Norm, Scale, Elementwise, BinOp
- Flash Attention (MHA, GQA, causal mask)
- Top-K, Embedding, KV-cache update
- PyTorch aten op dispatch (mm, addmm, linear, sdpa, layer_norm, softmax, ...)
- **Ollama full model offload on Windows (confirmed RX 9070 XT gfx1201)**

## What does not work yet

- Training (no autograd)
- Multi-GPU
- FP8 (requires `VK_KHR_shader_float8`, not yet in drivers)
- BFloat16 kernels
- `hipblaslt` workspace allocation hints
- Async compute (all dispatch is currently synchronous)

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
