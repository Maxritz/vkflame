# AGENTS.md
## vkflame — Automated Implementation Guide (Windows 11)

> Written for AI coding agents: Cursor, Cline, Aider, GitHub Copilot Workspace.
> Read fully before writing any code. Follow tasks in dependency order.
> Each task has an exact acceptance test. Do not mark done until the test passes.

---

## Before you start

Run these in PowerShell. All must pass.

```powershell
vulkaninfo --summary | Select-String "deviceName|apiVersion"
glslc --version
spirv-val --version
cmake --version
python --version
Write-Host "VULKAN_SDK: $env:VULKAN_SDK"
Write-Host "HIP_PATH:   $env:HIP_PATH"
```

If `VULKAN_SDK` is empty: install LunarG Vulkan SDK from https://vulkan.lunarg.com/sdk/home
If `HIP_PATH` is empty: install AMD HIP SDK from https://www.amd.com/en/developer/rocm-hub/hip-sdk.html

---

## Context files

| File | What to learn from it |
|---|---|
| `vkflame_architecture.md` | Full design spec — push constant layouts, descriptor bindings, kernel algorithms |
| `cdna_to_rdna4_migration_guide_v3.md` | What MFMA ops RDNA4 lacks, what WMMA ops it has instead |
| `amd_gpu_programming_guide.md` | LDS bank rules, wave32 reduce patterns |
| `runtime/device.cpp` | Reference implementation — pNext chain, feature detection, queue setup |
| `runtime/pipeline.cpp` | Reference implementation — push constant size table, pipeline creation |
| `runtime/dispatch.cpp` | Reference implementation — push constant structs, static_assert, descriptor binding |

Read the three runtime .cpp files before writing anything. They are the ground truth.

---

## Hard rules — these caused real crashes, do not violate them

**RULE 1 — VkPhysicalDeviceVulkan12Features is exclusive**

Use ONE struct for all Vulkan 1.2 features. Never chain individual extension structs
alongside `VkPhysicalDeviceVulkan12Features`. The combination violates
`VUID-VkDeviceCreateInfo-pNext-02830` and the device will fail to create.

```cpp
// CORRECT
VkPhysicalDeviceVulkan12Features vk12{};
vk12.sType                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
vk12.shaderFloat16           = VK_TRUE;
vk12.shaderInt8              = VK_TRUE;
vk12.storageBuffer8BitAccess = VK_TRUE;   // required by linear_int8.glsl
vk12.bufferDeviceAddress     = VK_TRUE;
vk12.pNext                   = nullptr;

// WRONG — do not also add any of these:
// VkPhysicalDeviceShaderFloat16Int8Features
// VkPhysicalDevice8BitStorageFeatures
// VkPhysicalDeviceBufferDeviceAddressFeatures
// ... any other struct that overlaps with Vk12
```

**RULE 2 — storageBuffer8BitAccess must be enabled**

`linear_int8.glsl` uses `StorageBuffer` with 8-bit integer elements.
Without `storageBuffer8BitAccess = VK_TRUE` in the Vk12 features struct,
`vkCreateShaderModule` fails with `VUID-RuntimeSpirv-storageBuffer8BitAccess-06328`
and all INT8 kernels silently produce zeros.

**RULE 3 — push constant range size must exactly match the shader**

The `VkPushConstantRange::size` in the pipeline layout must equal the size of the
GLSL `layout(push_constant)` block exactly. Too small → `VUID-VkComputePipelineCreateInfo-layout-10069`
→ `VK_ERROR_DEVICE_LOST` at dispatch time.

Every push constant struct in `dispatch.cpp` has a `static_assert` on its size.
The sizes in `pipeline.cpp`'s `k_pc_sizes[]` table must match those asserts.

```cpp
// dispatch.cpp pattern — copy this for every new shader
struct LinearPC {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    float    act_scale;
    uint32_t activation;
};
static_assert(sizeof(LinearPC) == 20, "LinearPC size must be 20");
```

**RULE 4 — subgroup size query**

RDNA4 (gfx1201) is wave32. The AMD Windows Vulkan driver reports `maxSubgroupSize=64`
via `VkPhysicalDeviceVulkan11Properties` but the actual execution size is 32.
Query `subgroupSize` (not `maxSubgroupSize`) from `VkPhysicalDeviceVulkan11Properties`
via `vkGetPhysicalDeviceProperties2`. See `device.cpp` for the correct query.

**RULE 5 — DLL interception uses PATH, not LD_PRELOAD**

LD_PRELOAD does not exist on Windows.
Our `hipblaslt.dll`, `hipblas.dll`, `amdhip64.dll` intercept ROCm calls by appearing
earlier in the PATH than `%HIP_PATH%\bin`.
`install.py` writes our build directory to the front of `HKCU\Environment\PATH` via winreg.
A new terminal must be opened after install for the change to take effect.

**RULE 6 — all exports need `__declspec(dllexport)`**

Use the `VKF_API` macro defined in `device.h` for every public function.
Without it, ctypes cannot find the symbol and Python gets a cryptic load error.

```cpp
#ifdef _WIN32
#  define VKF_API extern "C" __declspec(dllexport)
#else
#  define VKF_API extern "C"
#endif
```

---

## Global rules

**C++ rules:**
- C++17
- Every Vulkan call wrapped in `VKF_CHECK(call)` — prints file+line on failure
- No raw `new`/`delete` — use RAII or `std::unique_ptr`
- Include order: Windows headers first (`windows.h` with `WIN32_LEAN_AND_MEAN` and
  `NOMINMAX`), then `vulkan.h` with `VK_USE_PLATFORM_WIN32_KHR` already defined

**GLSL rules:**
- `#version 460` at top of every shader (not 450 — 1.3 extensions need 460)
- One `layout(push_constant)` block per shader
- No hardcoded sizes — use push constant fields or specialisation constants
- `shared float s_o[HEAD_DIM]` not `shared float s_o[64]` — use the spec constant

**Python rules:**
- Type hints everywhere
- `VKFLAME_DEBUG=1` triggers per-op stderr logging
- DLL paths use forward slashes — Python's ctypes handles Windows paths fine

**Verification rule:**
Write the test first. If the test fails, fix the implementation — never adjust the test.

---

## Push constant size reference

This table is the single source of truth. `pipeline.cpp` and `dispatch.cpp` must agree.

| Kernel | Fields | Size |
|---|---|---|
| `linear_int8` | M N K act_scale activation | 20 |
| `linear_fp16` | M N K activation | 16 |
| `linear_coop` | M N K activation | 16 |
| `flash_attention` | B Hq Hkv Sq Skv D scale is_causal | 32 |
| `winograd_f23` | OC IC H W | 16 |
| `winograd_f45` | OC IC H W | 16 |
| `conv_direct` | OC IC H W KH KW | 24 |
| `rms_norm` | M N eps | 12 |
| `softmax_online` | M N | 8 |
| `elementwise` | N op | 8 |
| `reduce` | M N op | 12 |
| `topk` | M N K largest | 16 |
| `sort_radix` | M N | 8 |
| `embedding` | V D B | 12 |
| `kvcache_update` | seq_pos n_heads head_dim max_seq | 16 |

---

## TASK 01 — CMakeLists.txt

**Write:** `CMakeLists.txt`

Requirements:
- `cmake_minimum_required(VERSION 3.20)`
- `project(vkflame LANGUAGES CXX)`
- `set(CMAKE_CXX_STANDARD 17)`
- `find_package(Vulkan REQUIRED)`
- `find_program(GLSLC glslc REQUIRED)`
- All targets get: `add_compile_definitions(VK_USE_PLATFORM_WIN32_KHR NOMINMAX WIN32_LEAN_AND_MEAN)`
- All .dll output to `build/` — no `Debug/` or `Release/` subdirs:
  ```cmake
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  foreach(config ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER ${config} CONFIG)
      set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG} ${CMAKE_BINARY_DIR})
      set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG} ${CMAKE_BINARY_DIR})
  endforeach()
  ```
- Custom target `spirv_kernels`: compiles all `.glsl` → `build/spirv/` then runs `embed_spirv.py`
- Shared libraries: `vkflame_rt`, `hipblaslt`, `hipblas`, `amdhip64`
- Test executables: `vkflame_tests`, `test_device`, `test_buffer`, `test_shim`
- `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)`

**Accept when:**
```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
# exits 0, prints "-- Found Vulkan:"
```

---

## TASK 02 — tools/embed_spirv.py

**Write:** `tools/embed_spirv.py`

Scans a directory of `.spv` files, emits a `.cpp` with:
- One `extern "C" __declspec(dllexport) const uint8_t vkf_spv_NAME[]` per shader
- One `const uint32_t vkf_spv_NAME_len` per shader
- A `vkf_spirv_table[]` lookup array with a `nullptr` sentinel at the end
- 16 hex bytes per line

**Accept when:**
```powershell
[System.IO.File]::WriteAllBytes("$env:TEMP\t.spv", [byte[]](0x03,0x02,0x23,0x07))
python tools/embed_spirv.py $env:TEMP build/spirv_embed.cpp
Select-String 'vkf_spirv_table' build/spirv_embed.cpp && Write-Host "PASS"
```

---

## TASK 03 — runtime/device.h

**Write:** `runtime/device.h`

Defines `VKF_API`, `VKFFeatures`, `VKFContext`, and declares the four public functions.
The enum `VKFKernelID` and `VKFPipeline` struct also live here — pipeline.cpp needs them.

```cpp
#pragma once
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  define VK_USE_PLATFORM_WIN32_KHR
#  include <windows.h>
#endif
#include <vulkan/vulkan.h>
#include <cstdint>

#ifdef _WIN32
#  define VKF_API extern "C" __declspec(dllexport)
#else
#  define VKF_API extern "C"
#endif

struct VKFFeatures {
    bool     has_cooperative_matrix;
    bool     has_float8;
    bool     has_integer_dot_product;
    bool     has_float16;
    bool     has_int8;
    bool     has_buffer_device_address;
    uint32_t subgroup_size;
    uint32_t max_workgroup_x;
    uint32_t max_compute_shared_memory;
    char     device_name[256];
    uint32_t vendor_id;
};

struct VKFContext {
    VkInstance       instance;
    VkPhysicalDevice physical_device;
    VkDevice         device;
    VkQueue          compute_queue;
    VkQueue          transfer_queue;
    uint32_t         compute_family;
    uint32_t         transfer_family;
    VkCommandPool    command_pool;
    VKFFeatures      features;
};

enum VKFKernelID {
    VKF_KERNEL_LINEAR_INT8      = 0,
    VKF_KERNEL_LINEAR_FP16      = 1,
    VKF_KERNEL_LINEAR_COOP      = 2,
    VKF_KERNEL_FLASH_ATTENTION  = 3,
    VKF_KERNEL_WINOGRAD_F23     = 4,
    VKF_KERNEL_WINOGRAD_F45     = 5,
    VKF_KERNEL_CONV_DIRECT      = 6,
    VKF_KERNEL_RMS_NORM         = 7,
    VKF_KERNEL_SOFTMAX_ONLINE   = 8,
    VKF_KERNEL_ELEMENTWISE      = 9,
    VKF_KERNEL_REDUCE           = 10,
    VKF_KERNEL_TOPK             = 11,
    VKF_KERNEL_SORT_RADIX       = 12,
    VKF_KERNEL_EMBEDDING        = 13,
    VKF_KERNEL_KVCACHE_UPDATE   = 14,
    VKF_KERNEL_COUNT            = 15
};

struct VKFPipeline {
    VkPipeline            pipeline;
    VkPipelineLayout      layout;
    VkDescriptorSetLayout ds_layout;
    uint32_t              push_constant_size;
};

VKF_API VKFContext* vkflame_get_context();
VKF_API int         vkflame_init();
VKF_API void        vkflame_shutdown();
VKF_API void        vkflame_print_info();

// pipeline.cpp
VKF_API int          vkflame_pipelines_init();
VKF_API VKFPipeline* vkflame_get_pipeline(VKFKernelID id);
VKF_API void         vkflame_pipelines_destroy();
```

---

## TASK 04 — runtime/device.cpp

**Do not rewrite.** The reference implementation is at `runtime/device.cpp`.
Copy it as-is. The three hard rules above were learned from its bugs — it is now correct.

If modifying: re-read RULE 1, RULE 2, RULE 4 before touching anything in `create_device()`
or `detect_features()`.

---

## TASK 05 — runtime/buffer.h and buffer.cpp

**Write:** `runtime/buffer.h`, `runtime/buffer.cpp`

```cpp
struct VKFBuffer {
    VkBuffer        buffer;
    VkDeviceMemory  memory;
    VkDeviceAddress address;
    size_t          size;
};

VKF_API VKFBuffer* vkflame_alloc(size_t size);
VKF_API void       vkflame_free(VKFBuffer* buf);
VKF_API VKFBuffer* vkflame_buf_from_ptr(void* ptr);
VKF_API int        vkflame_memcpy_h2d(VKFBuffer* dst, const void* src, size_t size, size_t offset);
VKF_API int        vkflame_memcpy_d2h(void* dst, VKFBuffer* src, size_t size, size_t offset);
VKF_API int        vkflame_memcpy_d2d(VKFBuffer* dst, VKFBuffer* src, size_t size);
VKF_API int        vkflame_memset(VKFBuffer* buf, int value, size_t size);
```

- `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` for GPU buffers
- Staging buffer for h2d/d2h transfers
- Global `std::unordered_map<VkDeviceAddress, VKFBuffer*>` with `std::mutex`
- 256-byte alignment on all allocations

**Accept when:**
```powershell
.\build\test_buffer.exe
# Must print: PASS: 1MB round-trip
```

---

## TASK 06 — runtime/pipeline.cpp

**Do not rewrite.** The reference implementation is at `runtime/pipeline.cpp`.
Copy it as-is. The `k_pc_sizes[]` table is correct and verified.

If adding a new shader: add its name to `k_kernel_names[]`, its size to `k_pc_sizes[]`,
and its `VKFKernelID` enum value — all three in the same position.

---

## TASK 07 — runtime/dispatch.cpp

**Do not rewrite.** The reference implementation is at `runtime/dispatch.cpp`.
Copy it as-is. All push constant structs have `static_assert` size checks.

If adding a new dispatch function:
1. Define the push constant struct with a `static_assert`
2. Verify its size matches `k_pc_sizes[]` in `pipeline.cpp`
3. Use `ceil_div()` for workgroup counts — never pass 0 to `vkCmdDispatch`
4. Create and destroy the descriptor pool within the same function call

---

## TASK 08 — shaders (kernels/*.glsl)

Write one shader at a time. After each one:
```powershell
glslc --target-env=vulkan1.3 -O -fshader-stage=compute kernels\NAME.glsl -o NUL
# exits 0 = compile OK

glslc --target-env=vulkan1.3 -O -fshader-stage=compute kernels\NAME.glsl -o build\spirv\NAME.spv
spirv-val build\spirv\NAME.spv
# both must succeed before moving to the next shader
```

### kernels/rms_norm.glsl

```glsl
#version 460
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_16bit_storage       : require

layout(local_size_x = 256) in;

layout(set=0, binding=0) readonly  buffer X     { float16_t x[]; };
layout(set=0, binding=1) readonly  buffer Gamma { float16_t g[]; };
layout(set=0, binding=2) writeonly buffer Y     { float16_t y[]; };

layout(push_constant) uniform PC { uint M; uint N; float eps; } pc;
// sizeof = 12 — matches k_pc_sizes[VKF_KERNEL_RMS_NORM]

shared float s_rrms;
```

Algorithm:
1. Sum-of-squares over N with subgroupAdd
2. `s_rrms = inversesqrt(sum / N + eps)` — broadcast via shared mem
3. `y[i] = float16_t(float(x[i]) * s_rrms * float(g[col]))`

Verify: `x=[1,2,3,4]` gamma=[1,1,1,1] eps=1e-6 → `[0.3651, 0.7303, 1.0954, 1.4606]`

### kernels/softmax_online.glsl

```glsl
layout(push_constant) uniform PC { uint M; uint N; } pc;
// sizeof = 8 — matches k_pc_sizes[VKF_KERNEL_SOFTMAX_ONLINE]
```

Online two-pass. Subgroup merge formula:
`m_new = max(ma, mb); d_new = da*exp(ma-m_new) + db*exp(mb-m_new)`
Do NOT use plain `subgroupAdd` — it gives wrong results for softmax.

### kernels/linear_int8.glsl

```glsl
layout(push_constant) uniform PC {
    uint  M; uint N; uint K; float act_scale; uint activation;
} pc;
// sizeof = 20 — matches k_pc_sizes[VKF_KERNEL_LINEAR_INT8]
```

- Extension: `GL_EXT_integer_dot_product`
- GELU activation = erf form: `0.5 * x * (1.0 + erf(x * 0.70710678))` — NOT tanh

### kernels/linear_fp16.glsl

```glsl
layout(push_constant) uniform PC {
    uint M; uint N; uint K; uint activation;
} pc;
// sizeof = 16 — matches k_pc_sizes[VKF_KERNEL_LINEAR_FP16]
```

### kernels/flash_attention.glsl

```glsl
layout(push_constant) uniform PC {
    uint B; uint Hq; uint Hkv; uint Sq; uint Skv; uint D; float scale; uint is_causal;
} pc;
// sizeof = 32 — matches k_pc_sizes[VKF_KERNEL_FLASH_ATTENTION]
```

- Specialisation constants: `layout(constant_id=0) const int HEAD_DIM = 64;`
- LDS: `shared float s_o[HEAD_DIM]` — NOT `shared float s_o[64]`
- GQA: `int kv_head = q_head / (Hq / Hkv);`
- This shader needs 4 bindings (Q K V O). Descriptor set layout in `pipeline.cpp`
  must have 4 bindings for this kernel. Check `build_pipeline()` — the binding count
  for `VKF_KERNEL_FLASH_ATTENTION` must be 4, not `MAX_BINDINGS=3`.

### kernels/topk.glsl, embedding.glsl, kvcache_update.glsl

Push constant sizes from the table above. All compile to `NUL` exits 0.

---

## TASK 09 — shim/hip_runtime_shim.cpp

**Write:** `shim/hip_runtime_shim.cpp`

Exports (all with `VKF_API`):
```
hipMalloc hipFree hipMallocManaged
hipMemcpy hipMemcpyAsync hipMemset hipMemsetAsync
hipDeviceSynchronize
hipStreamCreate hipStreamDestroy hipStreamSynchronize
hipGetDeviceProperties hipSetDevice hipGetDevice hipDeviceGetAttribute
hipGetErrorString hipGetLastError hipPeekAtLastError
```

`hipMalloc` → `vkflame_alloc` → store `ptr→VKFBuffer*` map → return `(void*)buf->address`
`hipMemcpy` kind 1=H2D, 2=D2H, 3=D2D → call matching `vkflame_memcpy_*`
`hipGetDeviceProperties` → fill from `VKFContext::features`
  - `warpSize = features.subgroup_size` (32 on gfx1201 after RULE 4 fix)
  - `gcnArchName = "gfx1201"` if `vendor_id == 0x1002`

**Accept when:**
```python
import ctypes
hip = ctypes.CDLL(r"build\amdhip64.dll")
ptr = ctypes.c_void_p()
assert hip.hipMalloc(ctypes.byref(ptr), 1024) == 0 and ptr.value != 0
hip.hipFree(ptr)
print("PASS")
```

---

## TASK 10 — shim/hipblas_shim.cpp

**Write:** `shim/hipblas_shim.cpp`

Exports: `hipblasCreate hipblasDestroy hipblasSgemm hipblasHgemm`
and batched/strided variants. `hipblasSgemm` → `vkflame_dispatch_linear`.

**Accept when:**
```python
import ctypes
blas = ctypes.CDLL(r"build\hipblas.dll")
h = ctypes.c_void_p()
assert blas.hipblasCreate(ctypes.byref(h)) == 0
blas.hipblasDestroy(h)
print("PASS")
```

---

## TASK 11 — shim/hipblaslt_shim.cpp

**Write:** `shim/hipblaslt_shim.cpp`

`hipblasLtMatmul` extracts M/N/K from layout descriptors, maps activation epilogue
constants to ints, calls `vkflame_dispatch_linear`. All other functions are stubs.

**Accept when:**
```python
import ctypes
lt = ctypes.CDLL(r"build\hipblaslt.dll")
h = ctypes.c_void_p()
assert lt.hipblasLtCreate(ctypes.byref(h)) == 0
lt.hipblasLtDestroy(h)
print("PASS")
```

---

## TASK 12 — torch_backend/mode.py and ops.py

No Windows-specific changes. Python is Python.

`mode.py`: `TorchDispatchMode` subclass, `_HANDLED_OPS` dict, `install()` function.
`ops.py`: handlers for all 20 ops, call `vkflame_dispatch_*` via ctypes.

DLL loading in `__init__.py` uses `.dll` suffix on Windows:
```python
_dll_name = "vkflame_rt.dll" if sys.platform == "win32" else "libvkflame_rt.so"
```

---

## TASK 13 — python/vkflame/install.py

Uses `winreg` to prepend build dir to `HKCU\Environment\PATH`.
Calls `SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, ...)` so running
Explorer picks up the change without a reboot.
`--undo` flag removes the entry.

Full implementation is in the project spec — copy it exactly.

---

## TASK 14 — tools/quantise.py and benchmark.py

No Windows-specific changes. Pure Python.

---

## TASK 15 — tests/test_correctness.py

The full test file is embedded below. Do not simplify. If a test fails, fix the
implementation. Never change test tolerances.

```python
import torch, pytest

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPU")

def setup_module():
    import vkflame
    vkflame.install(permanent=True)

def test_mm_f16():
    A = torch.randn(256, 256, dtype=torch.float16).cuda()
    B = torch.randn(256, 256, dtype=torch.float16).cuda()
    ref = torch.mm(A.float(), B.float()).half()
    out = torch.mm(A, B)
    assert (out.float() - ref.float()).abs().max() < 0.1

def test_mm_shapes():
    for M, N, K in [(1, 4096, 4096), (32, 4096, 4096), (4096, 4096, 128)]:
        A = torch.randn(M, K, dtype=torch.float16).cuda()
        B = torch.randn(K, N, dtype=torch.float16).cuda()
        ref = torch.mm(A.float(), B.float()).half()
        out = torch.mm(A, B)
        assert (out.float() - ref.float()).abs().max() < 0.5

def test_rms_norm():
    x = torch.randn(32, 4096, dtype=torch.float16).cuda()
    g = torch.ones(4096, dtype=torch.float16).cuda()
    rms = (x.float().pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
    ref = (x.float() / rms * g.float()).half()
    out, _, _ = torch.ops.aten.native_layer_norm(x, [4096], g, None, 1e-6)
    assert (out.float() - ref.float()).abs().max() < 0.05

def test_softmax_numerical():
    x = torch.randn(64, 32000, dtype=torch.float16).cuda()
    ref = torch.softmax(x.float(), dim=-1).half()
    out = torch.softmax(x, dim=-1)
    assert (out.float() - ref.float()).abs().max() < 1e-3

def test_softmax_sum_to_one():
    x = torch.randn(8, 1000, dtype=torch.float16).cuda()
    out = torch.softmax(x, dim=-1)
    assert (out.float().sum(dim=-1) - 1.0).abs().max() < 1e-3

def test_sdpa_mha():
    B, H, S, D = 2, 8, 64, 64
    Q = torch.randn(B, H, S, D, dtype=torch.float16).cuda()
    K = torch.randn(B, H, S, D, dtype=torch.float16).cuda()
    V = torch.randn(B, H, S, D, dtype=torch.float16).cuda()
    ref = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float()).half()
    out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    assert (out.float() - ref.float()).abs().max() < 0.1

def test_sdpa_causal():
    B, H, S, D = 1, 4, 32, 64
    Q = torch.randn(B, H, S, D, dtype=torch.float16).cuda()
    K = torch.randn(B, H, S, D, dtype=torch.float16).cuda()
    V = torch.randn(B, H, S, D, dtype=torch.float16).cuda()
    ref = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float(), is_causal=True).half()
    out = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    assert (out.float() - ref.float()).abs().max() < 0.1

def test_sdpa_gqa():
    B, Hq, Hkv, S, D = 1, 8, 2, 64, 64
    Q = torch.randn(B, Hq, S, D, dtype=torch.float16).cuda()
    K = torch.randn(B, Hkv, S, D, dtype=torch.float16).cuda()
    V = torch.randn(B, Hkv, S, D, dtype=torch.float16).cuda()
    K_exp = K.repeat_interleave(Hq // Hkv, dim=1)
    V_exp = V.repeat_interleave(Hq // Hkv, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(Q.float(), K_exp.float(), V_exp.float()).half()
    out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    assert (out.float() - ref.float()).abs().max() < 0.1

def test_winograd_f23():
    x = torch.randn(4, 8, 64, 64, dtype=torch.float32).cuda()
    w = torch.randn(16, 8, 3, 3, dtype=torch.float32).cuda()
    ref = torch.nn.functional.conv2d(x, w, padding=0)
    out = torch.nn.functional.conv2d(x, w, padding=0)
    assert (out - ref).abs().max() < 1e-3

def test_topk():
    x = torch.tensor([[3., 1., 4., 1., 5., 9., 2., 6.]]).cuda()
    vals, idx = torch.topk(x, k=3)
    assert vals[0, 0].item() == 9.0
    assert vals[0, 1].item() == 6.0
    assert vals[0, 2].item() == 5.0
    assert idx[0, 0].item() == 5
    assert idx[0, 1].item() == 7
    assert idx[0, 2].item() == 4

def test_argmax():
    x = torch.tensor([[1., 3., 2.]]).cuda()
    assert torch.argmax(x, dim=-1).item() == 1

def test_multinomial_distribution():
    probs = torch.tensor([[0.1, 0.2, 0.7]]).cuda()
    samples = torch.multinomial(probs, num_samples=10000, replacement=True)
    counts = torch.bincount(samples[0], minlength=3).float()
    freqs = counts / counts.sum()
    assert abs(freqs[2].item() - 0.7) < 0.05

def test_embedding():
    weight = torch.randn(100, 64, dtype=torch.float16).cuda()
    idx    = torch.tensor([0, 5, 99, 42]).cuda()
    ref    = torch.nn.functional.embedding(weight, idx)
    out    = torch.nn.functional.embedding(weight, idx)
    assert torch.allclose(out, ref)

def test_full_generation():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    import os
    model_path = os.environ.get("VKFLAME_TEST_MODEL", "")
    if not model_path:
        pytest.skip("Set VKFLAME_TEST_MODEL to a quantised model path")
    tok   = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        inp = tok("Hello", return_tensors="pt").input_ids.cuda()
        out = model.generate(inp, max_new_tokens=10)
    assert len(tok.decode(out[0])) > 5
```

---

## Dependency order

```
TASK 01  CMakeLists.txt
TASK 02  embed_spirv.py
   |
TASK 03  device.h              <- header, no code
TASK 04  device.cpp            <- copy reference, do not rewrite
TASK 05  buffer.h / buffer.cpp <- memory
TASK 06  pipeline.cpp          <- copy reference, do not rewrite
TASK 07  dispatch.cpp          <- copy reference, do not rewrite
   |
TASK 08  kernels/*.glsl        <- all shaders, parallel
   |
TASK 09  hip_runtime_shim      <- needs buffer
TASK 10  hipblas_shim          <- needs dispatch
TASK 11  hipblaslt_shim        <- needs dispatch
   |
TASK 12  mode.py / ops.py
TASK 13  install.py
TASK 14  quantise.py / benchmark.py
   |
TASK 15  test_correctness.py   <- needs everything
```

---

## What to do when a test fails

1. Run with validation layer first — it names the exact VUID:
   ```powershell
   $env:VK_INSTANCE_LAYERS="VK_LAYER_KHRONOS_validation"
   $env:VK_LOADER_DEBUG="error"
   python -m pytest tests/test_correctness.py -v -s
   ```

2. VUID-VkDeviceCreateInfo-pNext-02830 → RULE 1 violated in device.cpp
3. VUID-RuntimeSpirv-storageBuffer8BitAccess → RULE 2 violated — add `storageBuffer8BitAccess=VK_TRUE`
4. VUID-VkComputePipelineCreateInfo-layout-10069 → RULE 3 violated — push constant size wrong
5. VK_ERROR_DEVICE_LOST with no VUID → likely dispatch with workgroup=0, or wrong descriptor binding
6. test output all zeros → pipeline created but dispatch binding wrong — check binding indices
7. softmax sum != 1.0 → using plain `subgroupAdd` instead of the online merge formula
8. flash attention GQA wrong → using `q_head % Hkv` instead of `q_head / (Hq / Hkv)`
