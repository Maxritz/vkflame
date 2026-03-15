# SETUP_AND_TEST.md
## vkflame — Windows 11 Setup and Incremental Test Plan

Two separate stacks. Test them in order. Never mix until both work independently.

---

## LAYER 1 — Vulkan / C++ Runtime

Vulkan has nothing to do with Python. This entire layer builds and runs
without pip, PyTorch, or any Python package beyond the base interpreter
(which `embed_spirv.py` needs as a build tool).

### Required installs

**LunarG Vulkan SDK**
https://vulkan.lunarg.com/sdk/home — Windows installer, run as administrator.
Installs: `vulkaninfo`, `glslc`, `spirv-val`, `spirv-dis`, validation layer,
and sets `VULKAN_SDK` and updates system PATH automatically.

```powershell
# Verify — in a NEW terminal after install
$env:VULKAN_SDK                    # must not be empty
vulkaninfo --summary               # must show your GPU
glslc --version                    # must show version
spirv-val --version                # must show version
```

**AMD HIP SDK for Windows**
https://www.amd.com/en/developer/rocm-hub/hip-sdk.html
Installs HIP runtime, headers, sets `HIP_PATH`.

```powershell
$env:HIP_PATH                      # e.g. C:\Program Files\AMD\ROCm\6.2
```

This is what our shims replace. Our `hipblaslt.dll`, `hipblas.dll`, `amdhip64.dll`
must appear in PATH before `%HIP_PATH%\bin` for interception to work.

**CMake**
https://cmake.org/download — choose "Add to system PATH" during install.

```powershell
cmake --version     # must be >= 3.20
```

**LLVM / Clang (or Visual Studio 2022)**
https://github.com/llvm/llvm-project/releases — LLVM Windows installer.
Or install Visual Studio 2022 with "Desktop development with C++" workload.
Either compiler works. CMake finds whichever is available.

```powershell
clang++.exe --version    # if using LLVM
# or
cl.exe /?                # if using MSVC
```

---

### Layer 1 verification checklist

Run all of this in PowerShell before writing a single line of code.

```powershell
# GPU visible and Vulkan 1.3
vulkaninfo --summary | Select-String "deviceName|apiVersion"
# Expect: AMD Radeon RX 9070 XT, apiVersion = 1.3.x

# Features we specifically need
vulkaninfo --summary | Select-String "shaderFloat16|shaderInt8|bufferDeviceAddress|storageBuffer8BitAccess"

# glslc can target vulkan1.3
echo "" | glslc --target-env=vulkan1.3 -fshader-stage=compute - -o NUL 2>&1

# Shader tools present
spirv-val --version
spirv-dis --version

# SDK env vars set
Write-Host "VULKAN_SDK: $env:VULKAN_SDK"
Write-Host "HIP_PATH:   $env:HIP_PATH"

# Headers present
Test-Path "$env:VULKAN_SDK\Include\vulkan\vulkan.h"         # True
Test-Path "$env:HIP_PATH\include\hip\hip_runtime.h"        # True

# cmake and compiler
cmake --version
clang++.exe --version 2>$null; if (-not $?) { cl.exe /? 2>&1 | Select-Object -First 1 }
```

---

## LAYER 2 — Python / PyTorch Backend

Needed only for `torch_backend/`, `tools/quantise.py`, `benchmark.py`, and tests.
The Vulkan runtime does not call Python. The shims do not call Python.
Python calls the runtime via ctypes — that is the only bridge.

### Required installs

```powershell
# ROCm PyTorch — NOT the default CUDA build
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
# Adjust version: check Get-ChildItem "$env:HIP_PATH" to find your ROCm version

# Other packages — numpy also fixes the warning you saw in test output
pip install pytest pytest-timeout numpy transformers huggingface_hub
```

### Layer 2 verification

```powershell
# ROCm build confirmed
python -c "import torch; assert torch.version.hip is not None, 'WRONG: CUDA build installed'; print(f'ROCm {torch.version.hip}')"

# GPU visible to PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Must print: AMD Radeon RX 9070 XT

# numpy present (fixes the UserWarning in test output)
python -c "import numpy; print(f'numpy {numpy.__version__}')"
```

---

## PART 2 — Piece-by-Piece Test Plan

A failure at any stage is a hard stop. Fix it before proceeding.

---

### STAGE 1 — Shader compilation (no build system)

Fastest feedback loop. Test GLSL compiles before touching CMake.

```powershell
New-Item -ItemType Directory -Force -Path build\spirv | Out-Null

Get-ChildItem kernels\*.glsl | ForEach-Object {
    $spv = "build\spirv\$($_.BaseName).spv"
    glslc --target-env=vulkan1.3 -O -fshader-stage=compute $_.FullName -o $spv
    if ($LASTEXITCODE -eq 0) { Write-Host "OK:    $($_.Name)" }
    else                     { Write-Host "FAIL:  $($_.Name)"; exit 1 }
}

Get-ChildItem build\spirv\*.spv | ForEach-Object {
    spirv-val $_.FullName
    if ($LASTEXITCODE -eq 0) { Write-Host "VALID: $($_.Name)" }
    else                     { Write-Host "INVALID: $($_.Name)"; exit 1 }
}
```

Note: on Windows, test compile with `-o NUL` not `-o /dev/null`.

---

### STAGE 2 — embed_spirv.py

```powershell
python tools\embed_spirv.py build\spirv build\spirv_embed.cpp

Select-String 'vkf_spv_rms_norm'        build\spirv_embed.cpp; Write-Host "rms_norm OK"
Select-String 'vkf_spv_flash_attention'  build\spirv_embed.cpp; Write-Host "flash_attention OK"
Select-String 'vkf_spirv_table'         build\spirv_embed.cpp; Write-Host "table OK"
Select-String 'nullptr, nullptr, 0'     build\spirv_embed.cpp; Write-Host "sentinel OK"

clang++.exe -c build\spirv_embed.cpp -o $env:TEMP\spirv_embed.obj && Write-Host "C++ compile OK"
```

---

### STAGE 3 — CMake configure

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# Must print: "-- Found Vulkan: ..."
# Must exit 0

Test-Path build\compile_commands.json    # True
```

If Vulkan not found: check `$env:VULKAN_SDK`, restart terminal, retry.

---

### STAGE 4 — Build runtime only

```powershell
cmake --build build --target vkflame_rt -j

Test-Path build\vkflame_rt.dll    # True

dumpbin /EXPORTS build\vkflame_rt.dll | Select-String "vkflame_init|vkflame_alloc|vkflame_pipelines_init"
# Must show three lines
```

---

### STAGE 5 — Device init (first real Vulkan call)

```powershell
cmake --build build --target test_device -j
.\build\test_device.exe
# Must print: [vkflame] AMD Radeon RX 9070 XT  subgroup:32  ...
# Must print: PASS
```

If `subgroup:64` is printed instead of `subgroup:32`: RULE 4 in AGENTS.md was not followed.
Fix `detect_features()` in `device.cpp` to use `vk11props.subgroupSize` not `maxSubgroupSize`.

If `VK_ERROR_INITIALIZATION_FAILED`: check validation layer output —
```powershell
$env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"
$env:VK_LOADER_DEBUG    = "error"
.\build\test_device.exe 2>&1 | Select-String "VUID|error"
```

The most common cause is RULE 1 — chaining both `VkPhysicalDeviceVulkan12Features`
and individual extension structs. The VUID will say `pNext-02830`.

---

### STAGE 6 — Buffer round-trip

```powershell
cmake --build build --target test_buffer -j
.\build\test_buffer.exe
# Must print: PASS: 1MB round-trip
```

---

### STAGE 7 — Pipeline cache

```powershell
cmake --build build --target vkflame_tests -j
Measure-Command { .\build\vkflame_tests.exe }   # first run: < 5s
Measure-Command { .\build\vkflame_tests.exe }   # second run: < 200ms (cache hit)

Test-Path "$env:LOCALAPPDATA\vkflame\pipeline_cache.bin"    # True
```

If pipelines fail to build, run with validation:
```powershell
$env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"
.\build\vkflame_tests.exe 2>&1 | Select-String "VUID"
```

`VUID-VkComputePipelineCreateInfo-layout-10069` = push constant size wrong.
Check `k_pc_sizes[]` in `pipeline.cpp` against the GLSL `layout(push_constant)` block.

`VUID-RuntimeSpirv-storageBuffer8BitAccess` = RULE 2 violated.
Add `vk12.storageBuffer8BitAccess = VK_TRUE` to `device.cpp`.

---

### STAGE 8 — Single kernel dispatch (rms_norm)

The critical proof that the full path works: allocate → upload → dispatch → download → verify.

```powershell
cmake --build build --target test_rms_norm -j
.\build\test_rms_norm.exe
# Must print: output[0] = 0.365x (expected ~0.3651)
# Must print: PASS
```

If output is all zeros: the pipeline created but the descriptor binding is wrong.
Check that `VKF_KERNEL_RMS_NORM` bindings in dispatch match `layout(set=0, binding=N)` in the shader.

If `VK_ERROR_DEVICE_LOST`: run with validation to get the VUID. Almost always a
push constant size mismatch — the `static_assert(sizeof(RmsNormPC) == 12)` in
`dispatch.cpp` should have caught it at compile time if the struct is correct.

---

### STAGE 9 — Build shims

```powershell
cmake --build build --target hipblaslt hipblas amdhip64 -j

Test-Path build\hipblaslt.dll    # True
Test-Path build\hipblas.dll      # True
Test-Path build\amdhip64.dll     # True

dumpbin /EXPORTS build\amdhip64.dll  | Select-String "hipMalloc"
dumpbin /EXPORTS build\hipblas.dll   | Select-String "hipblasSgemm"
dumpbin /EXPORTS build\hipblaslt.dll | Select-String "hipblasLtMatmul"
```

---

### STAGE 10 — Shim smoke test

```powershell
python -c "
import ctypes
hip    = ctypes.CDLL(r'build\amdhip64.dll')
blas   = ctypes.CDLL(r'build\hipblas.dll')
blaslt = ctypes.CDLL(r'build\hipblaslt.dll')

ptr = ctypes.c_void_p()
assert hip.hipMalloc(ctypes.byref(ptr), 1024) == 0 and ptr.value != 0
hip.hipFree(ptr)
print('hipMalloc/hipFree: OK')

h = ctypes.c_void_p()
assert blas.hipblasCreate(ctypes.byref(h)) == 0
blas.hipblasDestroy(h)
print('hipblasCreate/Destroy: OK')

h = ctypes.c_void_p()
assert blaslt.hipblasLtCreate(ctypes.byref(h)) == 0
blaslt.hipblasLtDestroy(h)
print('hipblasLtCreate/Destroy: OK')
print('STAGE 10 PASS')
"
```

---

### STAGE 11 — Python package and PATH install

```powershell
pip install -e python\

python -m vkflame.install
# Open a NEW terminal, then:
$env:PATH -split ";" | Select-Object -First 5
# First entry must be the vkflame build directory

$env:PYTHONPATH = "python"
python -c "import vkflame; vkflame.info()"
# Must print: [vkflame] AMD Radeon RX 9070 XT  subgroup:32 ...
```

---

### STAGE 12 — PyTorch dispatch intercept

```powershell
$env:PYTHONPATH  = "python"
$env:VKFLAME_DEBUG = "1"

python -c "
import torch, vkflame
vkflame.install(permanent=True)
a = torch.ones(4, 4, dtype=torch.float16).cuda()
b = torch.ones(4, 4, dtype=torch.float16).cuda()
c = torch.mm(a, b)
assert c[0,0].item() == 4.0, f'Expected 4.0 got {c[0,0].item()}'
print('STAGE 12 PASS')
"
# With VKFLAME_DEBUG=1 must also print: [vkflame] mm
```

---

### STAGE 13 — Correctness suite

```powershell
$env:PYTHONPATH = "python"
python -m pytest tests\test_correctness.py -v --tb=long --timeout=60
```

Run individual tests when debugging:
```powershell
python -m pytest tests\test_correctness.py::test_mm_f16     -v --tb=long
python -m pytest tests\test_correctness.py::test_rms_norm   -v --tb=long
python -m pytest tests\test_correctness.py::test_sdpa_mha   -v --tb=long
python -m pytest tests\test_correctness.py::test_topk       -v --tb=long
```

Expected: 13 passed, `test_full_generation` skips unless `VKFLAME_TEST_MODEL` is set.

---

### STAGE 14 — Quantise a model

```powershell
python -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/phi-2', local_dir='C:/models/phi2')
"

python -m vkflame.tools.quantise C:\models\phi2 C:\models\phi2-int8

python -c "
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained('C:/models/phi2-int8')
found = sum(1 for _, mod in m.named_modules() if hasattr(mod, 'weight_int8'))
print(f'PASS: {found} quantised layers')
"
```

---

### STAGE 15 — Benchmark

```powershell
python tools\benchmark.py
# Must print a table of latencies without errors
```

---

## PART 3 — What Needs What

```
LAYER 1 (Vulkan / C++)
  Installs:   LunarG Vulkan SDK, AMD HIP SDK, CMake, LLVM or MSVC
  No Python packages needed beyond base Python for embed_spirv.py
  Stages 1-10

LAYER 2 (Python / PyTorch)
  Requires:   Layer 1 fully passing
  Installs:   pip install torch (ROCm wheel), pytest, numpy, transformers
  Stages 11-15
```

---

## PART 4 — Failure Reference

| Symptom | Cause | Fix |
|---|---|---|
| 14 tests skip "No GPU" | CUDA PyTorch installed, not ROCm | `pip install torch --index-url .../rocm6.2` |
| `torch.version.hip` is None | Same | Same |
| `VUID-pNext-02830` | Vk12Features + individual structs both chained | Remove individual structs, keep only Vk12Features |
| `VUID-storageBuffer8BitAccess` | Feature not enabled | Add `vk12.storageBuffer8BitAccess = VK_TRUE` |
| `VUID-layout-10069` | Push constant range too small | Check `k_pc_sizes[]` against GLSL PC block |
| `VK_ERROR_DEVICE_LOST` no VUID | `vkCmdDispatch(cb, 0, ...)` | Check `ceil_div` in dispatch.cpp |
| Output all zeros | Descriptor binding wrong | Check binding index matches GLSL `binding=N` |
| `subgroup:64` on gfx1201 | Using `maxSubgroupSize` | Use `vk11props.subgroupSize` instead |
| DLL not found at runtime | Our dir not first in PATH | Run `python -m vkflame.install`, open new terminal |
| `vkflame_init() rc=-3` | Device creation failed | Run with `VK_LAYER_KHRONOS_validation`, read VUID |
| Pipeline cache always cold | `%LOCALAPPDATA%\vkflame\` not writable | `mkdir $env:LOCALAPPDATA\vkflame` |
| NumPy UserWarning in pytest | numpy not installed | `pip install numpy` |
| Softmax sum != 1.0 | Wrong subgroup reduce | Use online merge formula, not `subgroupAdd` |
| Flash attention GQA wrong | Wrong kv_head formula | `q_head / (Hq / Hkv)` not `q_head % Hkv` |
