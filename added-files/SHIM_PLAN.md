# SHIM_PLAN.md
## vkflame — Complete ROCm Shim Plan

> The shims intercept ROCm API calls at the DLL boundary.
> They do NOT call through to the real ROCm library.
> They do NOT wrap AMD's math.
> They call vkflame's own Vulkan kernels — leaner, smaller, faster.
> The ROCm DLLs do not need to exist on the target machine.

---

## Core principle

```
App / PyTorch / MIOpen / ONNX Runtime
        |
        | calls rocblasSgemm, miopenConvolutionForward, etc.
        v
  vkflame shim DLL  (same name, same exports, same ABI)
        |
        | routes to
        v
  vkflame_rt.dll   +   Vulkan kernels (our own GLSL)
        |
        v
  RX 9070 XT via Vulkan 1.3
```

Nothing from AMD's stack runs. The shim is the complete replacement.

---

## DLL inventory — every file to create

| Shim DLL | Replaces | Phase |
|---|---|---|
| `amdhip64.dll` | HIP runtime | 1 — done |
| `hipblas.dll` | hipBLAS level-2/3 BLAS | 1 — done |
| `hipblaslt.dll` | hipBLASLt flexible GEMM | 1 — done |
| `rocblas.dll` | rocBLAS (lower-level than hipBLAS) | 2 |
| `MIOpen.dll` | MIOpen deep learning primitives | 2 |
| `hipfft.dll` | hipFFT (wraps rocFFT) | 2 |
| `rocfft.dll` | rocFFT internals | 2 |
| `hiprand.dll` | hipRAND / rocRAND PRNG | 2 |
| `hipsolver.dll` | hipSOLVER — LU/QR/SVD/eigensolver | 3 |
| `hipsparse.dll` | hipSPARSE — sparse linear algebra | 3 |
| `hipsparselt.dll` | hipSPARSELt — 2:4 sparse GEMM | 3 |
| `MIGraphX.dll` | MIGraphX — ONNX graph compiler | 4 |
| `migraphx_c.dll` | MIGraphX C API | 4 |
| `rocwmma.dll` | rocWMMA — wave matrix ops | 4 |

Phase 1 = inference critical, already partially done
Phase 2 = completes the standard AI inference stack
Phase 3 = scientific computing and sparse ops
Phase 4 = ONNX/graph-level interception (Amuse path)

---

## PHASE 1 — Already built (review and complete)

### amdhip64.dll (shim/hip_runtime_shim.cpp)

Status: written. Verify these symbols are all exported:

```
hipMalloc                hipFree                 hipMallocManaged
hipMemcpy               hipMemcpyAsync           hipMemcpyHtoD
hipMemcpyDtoH           hipMemcpyDtoD            hipMemset
hipMemsetAsync          hipDeviceSynchronize     hipGetLastError
hipPeekAtLastError      hipGetErrorString        hipSetDevice
hipGetDevice            hipGetDeviceCount        hipGetDeviceProperties
hipDeviceGetAttribute   hipStreamCreate          hipStreamDestroy
hipStreamSynchronize    hipStreamWaitEvent       hipEventCreate
hipEventDestroy         hipEventRecord           hipEventSynchronize
hipEventElapsedTime     hipLaunchKernelGGL       hipModuleLaunchKernel
hipModuleLoad           hipModuleGetFunction     hipModuleUnload
hipHostMalloc           hipHostFree              hipPointerGetAttributes
hipFuncSetAttribute     hipOccupancyMaxActiveBlocksPerMultiprocessor
```

Note: `hipModuleLaunchKernel` and `hipModuleLoad` must stub out cleanly —
return `hipSuccess` without executing anything. Triton and other JIT
compilers call these. They should not crash; they just won't run kernel code.

---

### hipblas.dll (shim/hipblas_shim.cpp)

All calls route to `vkflame_dispatch_linear`.

```
hipblasCreate           hipblasDestroy          hipblasSetStream
hipblasGetStream        hipblasSetMathMode

hipblasSgemm            hipblasHgemm            hipblasDgemm
hipblasCgemm            hipblasZgemm

hipblasSgemmBatched     hipblasHgemmBatched
hipblasSgemmStridedBatched  hipblasHgemmStridedBatched

hipblasSgemv            hipblasHgemv
hipblasSger             hipblasHger
hipblasSdot             hipblasHdot
hipblasSscal            hipblasHscal
hipblasSaxpy            hipblasHaxpy
hipblasSnrm2            hipblasHnrm2
hipblasSasum

hipblasStrsm            hipblasHtrsm
hipblasSsymm            hipblasHsymm
hipblasSsyrk            hipblasHsyrk
hipblasSsyr2k           hipblasHsyr2k
hipblasStrmm            hipblasHtrmm
```

Vulkan kernels needed (beyond what exists):
- `gemv_fp16.glsl` — matrix-vector (BLAS L2) via row-parallel reduction
- `axpy_fp16.glsl` — `y = alpha*x + y` (elementwise, already covered)
- `dot_fp16.glsl` — reduction to scalar
- `trsm_lower.glsl`, `trsm_upper.glsl` — triangular solve

---

### hipblaslt.dll (shim/hipblaslt_shim.cpp)

Status: written. Add these missing epilogue mappings:

```
HIPBLASLT_EPILOGUE_NONE         → activation=0
HIPBLASLT_EPILOGUE_RELU         → activation=2
HIPBLASLT_EPILOGUE_BIAS         → fused bias add (new: linear_bias.glsl)
HIPBLASLT_EPILOGUE_GELU         → activation=3 (erf form)
HIPBLASLT_EPILOGUE_GELU_BIAS    → gelu + bias fused
HIPBLASLT_EPILOGUE_SWISH_EXT    → activation=1 (SiLU)
HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT → silu + bias fused
```

New kernel needed:
- `linear_int8.glsl` already handles bias if passed through binding 2
- Bias should be a separate optional buffer at binding 2, zero-size = no bias

---

## PHASE 2 — Core inference completion

### rocblas.dll (shim/rocblas_shim.cpp)

rocBLAS is the lower-level backend that hipBLAS and MIOpen call internally.
If we shim rocblas.dll, MIOpen's GEMM path also gets captured.

```
rocblas_create_handle       rocblas_destroy_handle
rocblas_set_stream          rocblas_get_stream

rocblas_sgemm               rocblas_hgemm               rocblas_dgemm
rocblas_sgemm_batched       rocblas_hgemm_batched
rocblas_sgemm_strided_batched  rocblas_hgemm_strided_batched

rocblas_sgemv               rocblas_hgemv
rocblas_saxpy               rocblas_haxpy
rocblas_sdot                rocblas_hdot
rocblas_sscal               rocblas_hscal
rocblas_snrm2               rocblas_hnrm2
rocblas_sasum

rocblas_strsm               rocblas_htrsm
rocblas_strsv               rocblas_htrsv
rocblas_ssymm               rocblas_hsymm
rocblas_ssyrk               rocblas_hsyrk
rocblas_strmm               rocblas_htrmm

rocblas_sger                rocblas_hger
```

All route to `vkflame_dispatch_linear` or the appropriate L1/L2 dispatch.

---

### MIOpen.dll (shim/miopen_shim.cpp)

MIOpen is the deep learning primitives library. On Windows it ships with
ROCm 7.1/7.2 and is used by Amuse, ONNX Runtime AMD GPU EP, and PyTorch ROCm.

Intercept at the miopenConvolution* and miopenBatchNorm* level.
Our kernels replace MIOpen's auto-tuned solvers with fixed, lean Vulkan kernels.

```
// Handle management
miopenCreate                miopenDestroy
miopenSetStream             miopenGetStream
miopenCreateTensorDescriptor  miopenDestroyTensorDescriptor
miopenSet4dTensorDescriptor miopenGet4dTensorDescriptor
miopenSetTensorDescriptor   miopenGetTensorDescriptor

// Convolution
miopenCreateConvolutionDescriptor
miopenDestroyConvolutionDescriptor
miopenInitConvolutionDescriptor     // stride, pad, dilation
miopenConvolutionForward            // → winograd_f23.glsl or conv_direct.glsl
miopenConvolutionForwardBias        // → conv + bias_add.glsl
miopenConvolutionBackwardData       // → conv_backward_data.glsl   (Phase 3)
miopenConvolutionBackwardWeights    // → conv_backward_wt.glsl     (Phase 3)
miopenConvolutionForwardGetWorkSpaceSize
miopenFindConvolutionForwardAlgorithm   // return our one algorithm, always
miopenConvolutionForwardImmediate

// Batch normalisation
miopenCreateBatchNormDescriptor     // stub
miopenDestroyBatchNormDescriptor    // stub
miopenBatchNormalizationForwardInference   // → batch_norm_infer.glsl
miopenBatchNormalizationForwardTraining    // Phase 3
miopenBatchNormalizationBackward          // Phase 3

// Activation (already have all these in elementwise.glsl)
miopenCreateActivationDescriptor    miopenDestroyActivationDescriptor
miopenSetActivationDescriptor
miopenActivationForward             // → elementwise.glsl
miopenActivationBackward            // Phase 3

// Pooling
miopenCreatePoolingDescriptor       miopenDestroyPoolingDescriptor
miopenSet2dPoolingDescriptor        miopenGet2dPoolingDescriptor
miopenPoolingForward                // → pool_max.glsl or pool_avg.glsl
miopenPoolingBackward               // Phase 3

// Softmax (already done)
miopenSoftmaxForward                // → softmax_online.glsl
miopenSoftmaxBackward               // Phase 3

// Reduce
miopenReduceTensor                  // → reduce.glsl (already exists)

// RNN/LSTM — stub, return miopenStatusNotImplemented
miopenCreateRNNDescriptor
miopenDestroyRNNDescriptor
miopenRNNForward
miopenRNNBackward
```

New Vulkan kernels needed:
- `conv_direct.glsl` — fallback for non-3x3 or large-stride convolutions
- `batch_norm_infer.glsl` — fused mean/var/scale/shift, single-pass
- `pool_max.glsl` — max pooling with 2D sliding window
- `pool_avg.glsl` — average pooling
- `bias_add.glsl` — add bias vector to every row (can fold into elementwise)

Design note for conv routing:
```
miopenConvolutionForward:
  if filter == 3x3 and stride == 1 and dilation == 1:
      → winograd_f23.glsl  (already exists, fastest)
  elif filter == 1x1:
      → linear_fp16.glsl   (reshape to GEMM)
  else:
      → conv_direct.glsl   (general case)
```

---

### hipfft.dll + rocfft.dll (shim/hipfft_shim.cpp)

Critical for HammerForge science: astronomy, radio telescope, FFT convolution,
molecular dynamics, N-body. Our Stockham FFT is simpler and faster than
rocFFT's multi-pass plan compilation.

```
// Plan management
hipfftCreate                hipfftDestroy
hipfftSetStream
hipfftPlan1d                hipfftPlan2d                hipfftPlan3d
hipfftPlanMany              hipfftMakePlan1d            hipfftMakePlanMany
hipfftGetSize               hipfftGetSizeMany
hipfftEstimate1d            hipfftEstimate2d            hipfftEstimate3d

// Execution
hipfftExecC2C               // complex-to-complex → fft_c2c_1d.glsl
hipfftExecR2C               // real-to-complex   → fft_r2c_1d.glsl
hipfftExecC2R               // complex-to-real   → fft_c2r_1d.glsl
hipfftExecD2Z               // double precision  → stub or FP32 upcast
hipfftExecZ2D
hipfftExecZ2Z
```

New Vulkan kernels needed:
- `fft_c2c_1d.glsl` — Stockham radix-2 butterfly, in-place
- `fft_r2c_1d.glsl` — real FFT via half-size complex
- `fft_c2r_1d.glsl` — inverse
- `fft_twiddle.glsl` — precompute twiddle factors once per plan

Plan struct internally stores: N, batch, direction, twiddle buffer VkBuffer.

---

### hiprand.dll (shim/hiprand_shim.cpp)

Used by dropout, weight initialisation, sampling, Monte Carlo.
Philox 4x32 is the standard; Box-Muller converts to normal distribution.

```
hiprandCreateGenerator          hiprandDestroyGenerator
hiprandSetStream                hiprandSetGeneratorSeed
hiprandSetGeneratorOffset

hiprandGenerateUniform          // [0,1) float → prng_philox.glsl
hiprandGenerateUniformDouble    // → upcast from FP32
hiprandGenerateNormal           // N(mean,std) → prng_box_muller.glsl
hiprandGenerateNormalDouble
hiprandGenerateLogNormal
hiprandGenerateLogNormalDouble
hiprandGeneratePoisson          // → prng_poisson.glsl
hiprandGenerate                 // raw uint32

// quasi-random
hiprandCreateGeneratorHost      // stub — we don't need host-side PRNG
```

New kernels needed:
- `prng_philox.glsl` — 4x32 Philox counter-based PRNG, fills buffer
- `prng_box_muller.glsl` — transform uniform pairs to normal pairs
- `prng_poisson.glsl` — Poisson from normal approximation for large lambda

Philox is deterministic from a seed + counter. Push constants:
```glsl
layout(push_constant) uniform PC {
    uint seed;
    uint offset;   // counter start
    uint N;        // elements to generate
} pc;
```

---

## PHASE 3 — Scientific computing and sparse

### hipsolver.dll (shim/hipsolver_shim.cpp)

LU/QR/SVD/eigensolvers. Critical for molecular dynamics, physics simulation,
finite element methods. These are complex iterative algorithms — implement
the panel-blocked versions for good GPU utilisation.

```
hipsolverCreate             hipsolverDestroy            hipsolverSetStream

// LU factorisation (Gaussian elimination with partial pivoting)
hipsolverSgetrf             hipsolverDgetrf             // → lu_panel.glsl
hipsolverSgetrf_bufferSize  hipsolverDgetrf_bufferSize
hipsolverSgetrs             hipsolverDgetrs             // → trsm shaders

// QR factorisation (Householder)
hipsolverSgeqrf             hipsolverDgeqrf             // → householder.glsl
hipsolverSgeqrf_bufferSize
hipsolverSorgqr             hipsolverDorgqr             // → qr_apply_q.glsl

// SVD
hipsolverSgesvd             hipsolverDgesvd             // → svd_bidiag.glsl
hipsolverSgesvd_bufferSize

// Symmetric eigenvalue
hipsolverSsyevd             hipsolverDsyevd             // → tridiag_syevd.glsl
hipsolverSsyevd_bufferSize

// Linear solve (combines LU + GETRS)
hipsolverSgesv              hipsolverDgesv
```

New kernels needed:
- `lu_panel.glsl` — panel LU with partial pivoting
- `lu_update.glsl` — trailing matrix update after panel
- `householder.glsl` — Householder reflector computation
- `qr_apply_q.glsl` — apply stored Q to matrix
- `svd_bidiag.glsl` — bidiagonalisation (Golub-Reinsch)
- `tridiag_syevd.glsl` — divide-and-conquer symmetric eigensolver

These are the most mathematically complex kernels in the plan.
Implement in this order: getrf → getrs → geqrf → gesvd → syevd.
Each depends on trsm which is already in Phase 2.

---

### hipsparse.dll (shim/hipsparse_shim.cpp)

Sparse linear algebra. Used by sparse attention, graph neural networks,
scientific solvers with sparse matrices.

```
hipsparseCreate             hipsparseDestroy            hipsparseSetStream

// Matrix descriptors
hipsparseCreateMatDescr     hipsparseDestroyMatDescr
hipsparseSetMatType         hipsparseSetMatIndexBase

// Sparse matrix-vector (SpMV)
hipsparseScsrmv             hipsparseDcsrmv             // → spmv_csr.glsl
hipsparseSbsrmv             hipsparseDsbsrmv            // → spmv_bsr.glsl

// Sparse matrix-matrix (SpMM)
hipsparseScsrmm             hipsparseDcsrmm             // → spmm_csr.glsl
hipsparseScsrmm2            hipsparseDcsrmm2

// SpGEMM
hipsparseSpGEMM_createDescr hipsparseSpGEMM_destroyDescr
hipsparseSpGEMM_workEstimation
hipsparseSpGEMM_compute

// Format conversion
hipsparseScsr2csc           hipsparseDcsr2csc
hipsparseScsr2bsr           hipsparseDcsr2bsr
hipsparseDense2Csr          hipsparseCsr2Dense

// Generic API (newer ROCm)
hipsparseCreateSpVec        hipsparseDestroySpVec
hipsparseCreateCsr          hipsparseDestroySpMat
hipsparseSpMV               hipsparseSpMM
```

New kernels needed:
- `spmv_csr.glsl` — CSR format SpMV via warp-per-row
- `spmv_bsr.glsl` — BSR format SpMV via block-level WMMA
- `spmm_csr.glsl` — SpMM row merge strategy

---

### hipsparselt.dll (shim/hipsparselt_shim.cpp)

2:4 structured sparsity GEMM. We already have `v_swmmac_f32_16x16x32_f16`
mapped through vkflame's sparse GEMM kernel. This shim exposes the same
operation through the hipSPARSELt ABI.

```
hipsparseLtInit             hipsparseLtDestroy
hipsparseLtMatDescriptorInit
hipsparseLtMatmulDescriptorInit
hipsparseLtMatmulAlgSelectionInit
hipsparseLtMatmulPlanInit   hipsparseLtMatmulPlanDestroy
hipsparseLtMatmul           // → existing sparse_gemm.glsl
hipsparseLtSpMMAPrune       // → prune_2_4.glsl
hipsparseLtSpMMAPruneCheck
hipsparseLtSpMMACompress    // → compress_2_4.glsl
hipsparseLtSpMMACompressedSize
```

New kernels needed:
- `prune_2_4.glsl` — convert dense weights to 2:4 sparse
- `compress_2_4.glsl` — pack nonzeros + metadata into SWMMAC format

The actual sparse GEMM kernel already exists as part of HammerForge.

---

## PHASE 4 — Graph-level interception (Amuse / ONNX path)

### MIGraphX.dll + migraphx_c.dll (shim/migraphx_shim.cpp)

MIGraphX compiles ONNX graphs to optimised GPU programs.
Intercepting at this level means every model that runs through
Amuse, ONNX Runtime AMD GPU EP, or MIGraphX CLI runs through vkflame.

This is architecturally different from the other shims. Instead of
routing individual ops, we compile the ONNX graph ourselves using
our own Vulkan kernels, then cache the compiled version.

```
// Program management
migraphx_program_create         migraphx_program_destroy
migraphx_program_compile
migraphx_program_run
migraphx_program_get_output_shapes

// Context
migraphx_context_create         migraphx_context_destroy
migraphx_context_finish

// Parsing
migraphx_parse_onnx             // parse .onnx file → our graph IR
migraphx_parse_tf               // stub
migraphx_parse_torch            // stub

// Quantisation
migraphx_quantize_fp16
migraphx_quantize_int8

// C API wrappers
migraphx_load                   // load .mxr (compiled program)
migraphx_save                   // save .mxr
```

Implementation approach:
1. Parse ONNX with protobuf (onnx.proto already exists)
2. Walk the graph, for each node dispatch to our kernel:
   - MatMul → linear_fp16 or linear_int8
   - Conv → winograd_f23 or conv_direct
   - BatchNormalization → batch_norm_infer
   - Relu/Gelu/Silu → elementwise
   - Softmax → softmax_online
   - GlobalAveragePool → reduce
3. Cache compiled graph to `%LOCALAPPDATA%\vkflame\graph_cache\`

This is a substantial project on its own. Implement Phase 2+3 first.

---

## Kernel inventory — complete list

### Already exists (Phase 1 work)
```
linear_int8.glsl        — INT8 GEMM with OpSDot, fused activation
linear_fp16.glsl        — FP16 GEMM
linear_coop.glsl        — FP16 GEMM via cooperative matrix
flash_attention.glsl    — flash attention with GQA
winograd_f23.glsl       — 3x3 conv Winograd F(2,3)
winograd_filter_transform.glsl
winograd_f45.glsl       — 3x3 conv Winograd F(4,5)
conv_direct.glsl        — direct convolution fallback
rms_norm.glsl           — RMSNorm single-pass
softmax_online.glsl     — online two-pass softmax
elementwise.glsl        — relu/silu/gelu/add/mul/etc
reduce.glsl             — sum/max/mean reduction
topk.glsl               — top-k selection sort
sort_radix.glsl         — radix sort
embedding.glsl          — lookup table gather
kvcache_update.glsl     — KV cache in-place write
```

### Phase 2 additions
```
gemv_fp16.glsl          — matrix-vector product (BLAS L2)
axpy_fp16.glsl          — y = alpha*x + y (can extend elementwise)
dot_fp16.glsl           — dot product reduction to scalar
trsm_lower.glsl         — triangular solve, lower triangular
trsm_upper.glsl         — triangular solve, upper triangular
batch_norm_infer.glsl   — BN inference: (x - mean) / sqrt(var+eps) * gamma + beta
pool_max.glsl           — max pooling 2D sliding window
pool_avg.glsl           — average pooling 2D
bias_add.glsl           — broadcast bias vector addition
fft_c2c_1d.glsl         — Stockham complex-to-complex FFT
fft_r2c_1d.glsl         — real-to-complex FFT
fft_c2r_1d.glsl         — complex-to-real inverse FFT
fft_twiddle.glsl        — twiddle factor precompute
prng_philox.glsl        — Philox 4x32 uniform float PRNG
prng_box_muller.glsl    — uniform → normal transform
prng_poisson.glsl       — Poisson distribution sampling
```

### Phase 3 additions
```
spmv_csr.glsl           — sparse matrix-vector CSR
spmv_bsr.glsl           — sparse matrix-vector BSR
spmm_csr.glsl           — sparse matrix-matrix CSR
lu_panel.glsl           — LU panel factorisation
lu_update.glsl          — trailing matrix rank-k update
householder.glsl        — Householder reflector
qr_apply_q.glsl         — apply Q from QR to matrix
svd_bidiag.glsl         — bidiagonalisation
tridiag_syevd.glsl      — symmetric tridiagonal eigensolver
prune_2_4.glsl          — weight pruning to 2:4 sparsity
compress_2_4.glsl       — pack sparse data for SWMMAC
```

### Phase 4 additions
```
conv_backward_data.glsl      — gradient through conv (training)
conv_backward_wt.glsl        — gradient through conv weights
bn_backward.glsl             — batch norm backward
pool_backward.glsl           — pooling backward
activation_backward.glsl     — activation gradients
```

---

## Dispatch function additions needed (dispatch.cpp)

Add these new dispatch functions following the same pattern as the existing ones:

```cpp
// Phase 2
VKF_API void vkflame_dispatch_gemv(VKFContext*, ...);
VKF_API void vkflame_dispatch_dot(VKFContext*, ...);
VKF_API void vkflame_dispatch_trsm(VKFContext*, ..., int upper);
VKF_API void vkflame_dispatch_batch_norm_infer(VKFContext*, ...);
VKF_API void vkflame_dispatch_pool(VKFContext*, ..., int pool_type);
VKF_API void vkflame_dispatch_conv(VKFContext*, ...);   // routes to winograd/direct
VKF_API void vkflame_dispatch_fft(VKFContext*, ..., int direction);
VKF_API void vkflame_dispatch_prng_uniform(VKFContext*, ...);
VKF_API void vkflame_dispatch_prng_normal(VKFContext*, ...);

// Phase 3
VKF_API void vkflame_dispatch_spmv(VKFContext*, ..., int format);
VKF_API void vkflame_dispatch_lu(VKFContext*, ...);
VKF_API void vkflame_dispatch_qr(VKFContext*, ...);
VKF_API void vkflame_dispatch_svd(VKFContext*, ...);
```

Each follows the same pattern already in dispatch.cpp:
push constants → descriptor set → vkCmdDispatch → wait.

---

## Push constant size reference additions

Add to `k_pc_sizes[]` in pipeline.cpp (VKFKernelID enum must also expand):

| New kernel | Fields | Size |
|---|---|---|
| `gemv_fp16` | M N | 8 |
| `dot_fp16` | N | 4 |
| `trsm_lower` | N nrhs | 8 |
| `trsm_upper` | N nrhs | 8 |
| `batch_norm_infer` | N C H W eps | 20 |
| `pool_max` | N C OH OW KH KW SH SW PH PW | 44 |
| `pool_avg` | N C OH OW KH KW SH SW PH PW | 44 |
| `bias_add` | M N | 8 |
| `fft_c2c_1d` | N batch direction log2N | 16 |
| `fft_r2c_1d` | N batch log2N | 12 |
| `fft_c2r_1d` | N batch log2N | 12 |
| `prng_philox` | seed offset N | 12 |
| `prng_box_muller` | seed offset N mean stddev | 20 |
| `spmv_csr` | M N nnz | 12 |
| `lu_panel` | N nb offset | 12 |

---

## Math design notes — lean replacements

**Convolution:**
Standard: im2col + GEMM (O(N*K²*C) memory)
vkflame: Winograd F(2,3) for 3×3 (4 multiplies instead of 9), direct for rest.
No im2col buffer. No reshaping. Just tile-level register math.

**Batch normalisation (inference):**
Standard: separate mean/var pass, then normalise pass (2 global reads).
vkflame: single-pass fused kernel — one read, compute running stats in
shared memory within workgroup, apply scale/shift, one write.

**FFT:**
Standard: rocFFT plans, multi-pass algorithm selection, JIT compile.
vkflame: Stockham algorithm, fixed radix-2 butterfly, no JIT.
One kernel per transform direction. Plan = push constant struct.
Twiddle factors pre-computed once into a GPU buffer on plan creation.

**PRNG:**
Standard: rocRAND's Mersenne Twister or XORWOW (complex state management).
vkflame: Philox 4x32 — stateless, counter-based, trivially parallel.
Every thread takes its counter = global_id + offset, runs 10 rounds, outputs.
No state. No memory barrier. Perfect GPU parallelism.

**Sparse GEMM (2:4):**
Standard: hipSPARSELt with Tensile backend, offline tuning required.
vkflame: direct `v_swmmac_f32_16x16x32_f16` dispatch on gfx1201.
Prune + compress offline once per model (prune_2_4.glsl + compress_2_4.glsl).

---

## Windows PATH interception order

For full stack interception, PATH must be:
```
vkflame\build
%HIP_PATH%\bin     ← our DLLs shadow these
%SystemRoot%\System32
```

After `python -m vkflame.install`, the registry entry ensures this order.

If any ROCm DLL is NOT shimmed yet (e.g. migraphx in Phase 4), the real
DLL from `%HIP_PATH%\bin` is loaded as fallback. This is intentional —
unshimmed libraries still work via native ROCm. Shims are additive.

---

## Implementation order for agent

Follow this exact sequence. Each step is independently testable.

```
PHASE 2 SEQUENCE
  1. gemv_fp16.glsl        + dispatch function
  2. trsm_lower/upper.glsl + dispatch function
  3. rocblas_shim.cpp      test: dumpbin /EXPORTS build\rocblas.dll
  4. batch_norm_infer.glsl + dispatch function
  5. pool_max/avg.glsl     + dispatch function
  6. bias_add.glsl         (extend elementwise push constants)
  7. conv_direct.glsl      + update vkflame_dispatch_conv router
  8. miopen_shim.cpp       test: miopenConvolutionForward stub returns OK
  9. fft_c2c_1d.glsl       + twiddle precompute
  10. fft_r2c/c2r_1d.glsl
  11. hipfft_shim.cpp      test: hipfftExecC2C on 1024-point sine wave
  12. prng_philox.glsl     + box_muller.glsl
  13. hiprand_shim.cpp     test: GenerateUniform output in [0,1)

PHASE 3 SEQUENCE
  14. spmv_csr.glsl        + hipsparse_shim.cpp
  15. prune_2_4.glsl       + compress_2_4.glsl
  16. hipsparselt_shim.cpp test: sparse GEMM matches dense within 0.5%
  17. lu_panel.glsl        + lu_update.glsl
  18. trsm already done    (needed by getrs)
  19. householder.glsl     + qr_apply_q.glsl
  20. hipsolver_shim.cpp   test: getrf/getrs on 64x64 random matrix
  21. svd_bidiag.glsl      + tridiag_syevd.glsl
  22. extend hipsolver     with syevd

PHASE 4 SEQUENCE
  23. ONNX parser integration
  24. migraphx_shim.cpp
```

---

## Acceptance tests for each shim

### rocblas
```python
import ctypes
rb = ctypes.CDLL(r"build\rocblas.dll")
h = ctypes.c_void_p()
assert rb.rocblas_create_handle(ctypes.byref(h)) == 0
rb.rocblas_destroy_handle(h)
print("rocblas: PASS")
```

### MIOpen
```python
import ctypes
mi = ctypes.CDLL(r"build\MIOpen.dll")
h = ctypes.c_void_p()
assert mi.miopenCreate(ctypes.byref(h)) == 0
mi.miopenDestroy(h)
print("MIOpen: PASS")
```

### hipfft (numerical)
```python
import ctypes, numpy as np
# Create 1024-element complex array, execute C2C FFT, verify against numpy FFT
# Max relative error must be < 1e-4
```

### hiprand (distribution)
```python
import ctypes, numpy as np
# Generate 1M uniform floats, verify all in [0,1)
# Verify mean is 0.5 ± 0.001
# Verify std is 1/sqrt(12) ± 0.001
```

### hipsolver (numerical)
```python
import ctypes, numpy as np
# 64x64 random matrix A
# LU factorize, solve Ax=b
# Verify ||Ax - b|| / ||b|| < 1e-5
```
