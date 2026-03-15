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
- [Ollama for Windows](https://ollama.com/download/windows) — the AMD/ROCm bundle version
- The vkflame DLLs (download from [Releases](../../releases))

---

## Step 1 — Back up original DLLs

Before replacing anything, back up the originals.

```powershell
$rocmDir = "$env:LOCALAPPDATA\AMD\AI_Bundle\Ollama\lib\ollama\rocm"
$backup  = "$rocmDir\backup_original"
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
$rocmDir  = "$env:LOCALAPPDATA\AMD\AI_Bundle\Ollama\lib\ollama\rocm"
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

## Step 4 — Run a model

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
C:\TR\.venv\Scripts\python.exe -c "
import pefile
p = pefile.PE(r'$env:LOCALAPPDATA\AMD\AI_Bundle\Ollama\lib\ollama\rocm\amdhip64_6.dll')
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
$rocmDir = "$env:LOCALAPPDATA\AMD\AI_Bundle\Ollama\lib\ollama\rocm"
$backup  = "$rocmDir\backup_original"
Copy-Item "$backup\amdhip64_6.dll" $rocmDir -Force
Copy-Item "$backup\hipblas.dll"    $rocmDir -Force
Copy-Item "$backup\hipblaslt.dll"  $rocmDir -Force
Write-Host "Restored originals."
```
