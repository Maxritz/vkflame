C:\TR\vkflame> $py = "$env:LOCALAPPDATA\AMD\AI_Bundle\ComfyUI\venv\Scripts\python.exe"; Set-Location C:\TR\vkflame; & $py tools\gpu_bench.py 2>tools\bench_err.txt; Write-Host "Exit: $LASTEXITCODE"
[vkflame] runtime loaded OK  (C:\TR\vkflame\build\Release\vkflame_rt.dll)
================================================================================
  vkflame GPU Benchmark — ROCm vs Vulkan shime\buffer.cpp:129
  GPU    : AMD Radeon RX 9070 XTkflame\runtime\buffer.cpp:223
  PyTorch: 2.9.0+rocmsdk20251116kflame\runtime\buffer.cpp:129
  ROCm   : 7.1.52802-561cc400e1vkflame\runtime\buffer.cpp:223
  vkflame: YESsult=-4 at C:\TR\vkflame\runtime\buffer.cpp:129
  NOTE: Windows -- VKF timings include CPU<->Vulkan staging cost
        (HIP and Vulkan use separate VA spaces on Windows WDDM).
  STATUS: PRELIMINARY -- kernels correct, Windows overhead not yet zero-copy.
  BLOCKER: hipIpcGetMemHandle on PyTorch caching-allocator memory returns
           hipSuccess but triggers VK_ERROR_DEVICE_LOST in vkAllocateMemory.
           Enable opt-in with VKFLAME_WIN32_ZEROCOPY=1 once allocator is patched.
  PATH: vkflame_wrap_hip_ptr (VK_KHR_external_memory_win32) is implemented;
================================================================================

── FP16 GEMM (torch.mm) ─────────────────────────────────

op                    shape                           ROCm ms   VKF ms  speedup TFLOPS(vkf)   max_err
-----------------------------------------------------------------------------------------------------
mm_fp16               (256,256,256)                     0.038    3.403    0.01x         0.0  0.00e+00
mm_fp16               (1024,1024,1024)                  0.082    6.234    0.01x         0.3  0.00e+00
mm_fp16               (4096,4096,4096)                  1.164   47.530    0.02x         2.9  0.00e+00
mm_fp16               (1,4096,4096)                     0.106   19.567    0.01x         0.0  0.00e+00
mm_fp16               (32,4096,4096)                    0.089   16.903    0.01x         0.1  0.00e+00
mm_fp16               (128,8192,8192)                   0.260   59.052    0.00x         0.3  0.00e+00

── Batched FP16 GEMM (torch.bmm) ───────────────────────

op                    shape                           ROCm ms   VKF ms  speedup TFLOPS(vkf)   max_err
-----------------------------------------------------------------------------------------------------
bmm_fp16              (bs=4,1024,1024,1024)             0.112    0.135    0.83x        63.7  0.00e+00
bmm_fp16              (bs=32,128,4096,128)              0.133    0.163    0.82x        26.4  0.00e+00

── Flash Attention (scaled_dot_product_attention) ───────

op                    shape                           ROCm ms   VKF ms  speedup TFLOPS(vkf)   max_err
-----------------------------------------------------------------------------------------------------
sdpa_MHA              B1H32/32S512D64                   0.106    0.142    0.75x        15.1  0.00e+00
sdpa_MHA              B1H32/32S2048D64+causal           1.105    1.118    0.99x        30.7  0.00e+00
sdpa_MHA              B4H32/32S1024D128+causal          1.788    1.769    1.01x        38.9  0.00e+00
sdpa_GQA              B1H32/8S2048D64+causal            1.169    1.189    0.98x        28.9  0.00e+00
sdpa_GQA              B1H32/4S4096D64+causal            3.440    3.508    0.98x        39.2  0.00e+00

── RMS Norm ─────────────────────────────────────────────

op                    shape                           ROCm ms   VKF ms  speedup TFLOPS(vkf)   max_err
-----------------------------------------------------------------------------------------------------
rms_norm              (32,4096)                         0.027    4.817    0.01x         -    0.00e+00
rms_norm              (32,8192)                         0.023    3.848    0.01x         -    0.00e+00
rms_norm              (32,32768)                        0.034    4.933    0.01x         -    0.00e+00
rms_norm              (128,4096)                        0.019    3.819    0.00x         -    0.00e+00

── Softmax ──────────────────────────────────────────────

op                    shape                           ROCm ms   VKF ms  speedup TFLOPS(vkf)   max_err
-----------------------------------------------------------------------------------------------------
softmax               (1,32000)                         0.020    2.288    0.01x         -    0.00e+00
softmax               (32,32000)                        0.014    4.536    0.00x         -    0.00e+00
softmax               (32,128000)                       0.030   11.003    0.00x         -    0.00e+00
softmax               (128,32000)                       0.024   10.627    0.00x         -    0.00e+00

── Activations (element-wise unary) ─────────────────────

op                    shape                           ROCm ms   VKF ms  speedup TFLOPS(vkf)   max_err
-----------------------------------------------------------------------------------------------------
silu                  (n=131072)                        0.012    3.567    0.00x         -    0.00e+00
gelu                  (n=131072)                        0.007    3.582    0.00x         -    0.00e+00
relu                  (n=131072)                        0.011    3.427    0.00x         -    0.00e+00

── Element-wise Binary Ops ──────────────────────────────

op                    shape                           ROCm ms   VKF ms  speedup TFLOPS(vkf)   max_err
-----------------------------------------------------------------------------------------------------
add                   (n=524288)                        0.011    5.783    0.00x         -    0.00e+00
mul                   (n=524288)                        0.014    5.778    0.00x         -    0.00e+00
sub                   (n=524288)                        0.016    6.131    0.00x         -    0.00e+00
div                   (n=524288)                        0.038    6.020    0.01x         -    0.00e+00

── Top-K Sampling ───────────────────────────────────────

op                    shape                           ROCm ms   VKF ms  speedup TFLOPS(vkf)   max_err
-----------------------------------------------------------------------------------------------------
topk                  (1,32000,k=50)                    0.074    3.433    0.02x         -    0.00e+00
topk                  (32,32000,k=50)                   0.165    3.894    0.04x         -    0.00e+00
topk                  (1,128000,k=100)                  0.086    3.363    0.03x         -    0.00e+00

── Embedding Lookup ─────────────────────────────────────

op                    shape                           ROCm ms   VKF ms  speedup TFLOPS(vkf)   max_err
-----------------------------------------------------------------------------------------------------
embedding             (B=512,V=32000,D=256)             0.029    7.900    0.00x         -    0.00e+00
embedding             (B=2048,V=32000,D=256)            0.018    8.321    0.00x         -    0.00e+00
embedding             (B=512,V=128000,D=256)            0.016   27.034    0.00x         -    0.00e+00

── Full Transformer Layer (7B config, bs=1, seq=512) ─────

op                    shape                           ROCm ms   VKF ms  speedup TFLOPS(vkf)   max_err
-----------------------------------------------------------------------------------------------------
transformer_layer     bs=1,seq=512,7B                   2.641  390.555    0.01x         0.6      n/a 


══ Correctness spot-checks ══════════════════════════════
  PASS  mm_fp16 256x256                max_err=3.12e-02  tol=5e-01
  PASS  rms_norm 32x4096               max_err=0.00e+00  tol=5e-02
  PASS  softmax 4x32000                max_err=0.00e+00  tol=1e-03
  PASS  softmax_sums_to_1              max_row_err=8.34e-06
  PASS  sdpa_mha B2H8S64D64            max_err=4.88e-04  tol=1e-01
  PASS  sdpa_causal B1H4S32D64         max_err=9.77e-04  tol=1e-01
  PASS  topk [3,1,4,1,5,9,2,6] k=3     vals=[9,6,5] idx=[5,7,4]
  PASS  silu n=1024                    max_err=0.00e+00  tol=1e-02
  PASS  binop_add n=4096               max_err=0.00e+00  tol=1e-02
  PASS  binop_mul n=4096               max_err=0.00e+00  tol=1e-02
  PASS  binop_sub n=4096               max_err=0.00e+00  tol=1e-02

  All correctness checks PASSED.

Done.
Exit: 0