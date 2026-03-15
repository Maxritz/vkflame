# Installing vkflame for Ollama on Windows

This guide gets Ollama running with full GPU acceleration on any Vulkan 1.3 GPU
(AMD RDNA2+, NVIDIA Turing+, Intel Xe) without a real ROCm installation.

---

## What you need

- Windows 10 or 11
- A GPU with Vulkan 1.3 support
  - AMD: RX 6000 series (RDNA2) or newer
  - NVIDIA: RTX 20 series or newer
  - Intel: Arc A-series or newer
- [Ollama for Windows](https://ollama.com/download/windows) — standard installer **or** AMD bundle, both work
- The vkflame DLLs (download from [Releases](../../releases))

---

## Finding your Ollama rocm folder

Ollama installs to different locations depending on which package you used.
Run this once to find yours:

```powershell
$candidates = @(
    "$env:LOCALAPPDATA\Programs\Ollama\lib\ollama\rocm",
    "$env:LOCALAPPDATA\AMD\AI_Bundle\Ollama\lib\ollama\rocm"
)
$rocmDir = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($rocmDir) { Write-Host "Found: $rocmDir" } else { Write-Host "Not found — is Ollama installed?" }
```

Use the printed path in all the steps below.

---

## Step 1 — Back up original DLLs

Before replacing anything, back up the originals.

```powershell
# Auto-detect path (or set manually)
$rocmDir = @(
    "$env:LOCALAPPDATA\Programs\Ollama\lib\ollama\rocm",
    "$env:LOCALAPPDATA\AMD\AI_Bundle\Ollama\lib\ollama\rocm"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

$backup = "$rocmDir\backup_original"
New-Item -ItemType Directory -Force -Path $backup
Copy-Item "$rocmDir\amdhip64_6.dll" $backup
Copy-Item "$rocmDir\hipblas.dll"    $backup
Copy-Item "$rocmDir\hipblaslt.dll"  $backup
Write-Host "Backup saved to $backup"
```

---

## Step 2 — Copy vkflame DLLs

Extract `vkflame-windows-x64.zip` and copy the four files:

```powershell
# Auto-detect path (or set manually)
$rocmDir  = @(
    "$env:LOCALAPPDATA\Programs\Ollama\lib\ollama\rocm",
    "$env:LOCALAPPDATA\AMD\AI_Bundle\Ollama\lib\ollama\rocm"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
$vkflame  = "C:\path\to\extracted\vkflame-windows-x64"

Copy-Item "$vkflame\amdhip64_6.dll"  $rocmDir -Force
Copy-Item "$vkflame\hipblas.dll"     $rocmDir -Force
Copy-Item "$vkflame\hipblaslt.dll"   $rocmDir -Force
Copy-Item "$vkflame\vkflame_rt.dll"  $rocmDir -Force

Write-Host "Done."
```

You can also paste them manually in Explorer.

---

## Step 3 — Verify install

Run Ollama and check the startup log:

```
ollama serve
```

Success looks like:

```
inference compute id=0 library=ROCm compute=gfx1201 name=ROCm0
description="AMD Radeon RX 9070 XT" total="15.9 GiB" available="14.5 GiB"
```

Key things to confirm:

- `library=ROCm` — not `library=cpu`
- `compute=gfxXXXX` — your GPU arch is recognised
- `total=` shows your actual VRAM

---

## Step 4 — Set GPU arch for non-RDNA4 GPUs

vkflame defaults to `gfx1201` (RDNA4, RX 9070/9070 XT). If you have a different GPU,
set `HSA_OVERRIDE_GFX_VERSION` before launching Ollama:

| GPU                                  | Setting                           |
| ------------------------------------ | --------------------------------- |
| RX 9070 / 9070 XT (RDNA4)            | Nothing needed                    |
| RX 7900 / 7800 / 7700 / 7600 (RDNA3) | `HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| RX 6900 / 6800 / 6700 / 6600 (RDNA2) | `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| RX 6500 / 6400 (RDNA2 lite)          | `HSA_OVERRIDE_GFX_VERSION=10.1.3` |
| Explicit override                    | `VKFLAME_GFX_ARCH=gfxXXXX`        |

Set it in PowerShell before starting Ollama:

```powershell
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
ollama serve
```

Or add it permanently in: System Properties → Advanced → Environment Variables.

---

## Step 5 — Run a model

```
ollama run deepseek-r1:14b
```

With 16 GB VRAM the full Q4_K_M (8.4 GiB) fits with room for KV cache.
For smaller GPUs use a smaller quantisation (`deepseek-r1:7b`, `:1.5b`, etc.).

---

## Troubleshooting

### Still shows `library=cpu`

Check DLL exports — Windows won't load a DLL that's missing required exports:

```powershell
$rocmDir = @(
    "$env:LOCALAPPDATA\Programs\Ollama\lib\ollama\rocm",
    "$env:LOCALAPPDATA\AMD\AI_Bundle\Ollama\lib\ollama\rocm"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
python -c "
import pefile
p = pefile.PE(r'$rocmDir\amdhip64_6.dll')
exp = len(p.DIRECTORY_ENTRY_EXPORT.symbols) if hasattr(p,'DIRECTORY_ENTRY_EXPORT') else 0
print(f'amdhip64_6.dll exports: {exp}')
"
```

Should print `exports: 481` (or more). If it shows `0`, you have the old DLL still.

### Ollama crashes immediately

Run `ollama serve` from a terminal (not the tray icon) to see full stderr output.
Common causes:

- Missing `vkflame_rt.dll` in the rocm directory (all 4 must be present together)
- Vulkan driver not installed — run `vulkaninfo` to verify

### GPU not detected at all

```powershell
vulkaninfo --summary
```

Must show your GPU. If it doesn't, install/update the graphics driver.

---

## Reverting to original DLLs

```powershell
$rocmDir = @(
    "$env:LOCALAPPDATA\Programs\Ollama\lib\ollama\rocm",
    "$env:LOCALAPPDATA\AMD\AI_Bundle\Ollama\lib\ollama\rocm"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
$backup  = "$rocmDir\backup_original"
Copy-Item "$backup\amdhip64_6.dll" $rocmDir -Force
Copy-Item "$backup\hipblas.dll"    $rocmDir -Force
Copy-Item "$backup\hipblaslt.dll"  $rocmDir -Force
Write-Host "Restored originals."
```
