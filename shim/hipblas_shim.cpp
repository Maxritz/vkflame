// hipblas_shim.cpp — vkflame drop-in for hipBLAS API
// Routes sgemm/hgemm/gemmex to vkflame_dispatch_linear.

#include "../runtime/device.h"
#include "../runtime/dispatch.h"
#include <cstdio>
#include <cstdlib>

// ── hipBLAS types ─────────────────────────────────────────────────
typedef int hipblasStatus_t;
typedef void *hipStream_t;

#define HIPBLAS_STATUS_SUCCESS 0
#define HIPBLAS_STATUS_NOT_INITIALIZED 1
#define HIPBLAS_STATUS_ALLOC_FAILED 2
#define HIPBLAS_STATUS_INVALID_VALUE 3
#define HIPBLAS_STATUS_ARCH_MISMATCH 4
#define HIPBLAS_STATUS_MAPPING_ERROR 5
#define HIPBLAS_STATUS_EXECUTION_FAILED 6
#define HIPBLAS_STATUS_INTERNAL_ERROR 7
#define HIPBLAS_STATUS_NOT_SUPPORTED 8

// Operation type
#define HIPBLAS_OP_N 111
#define HIPBLAS_OP_T 112
#define HIPBLAS_OP_C 113

// Data type enum values (ROCm 6.x ABI)
typedef int hipblasDatatype_t;
#define HIPBLAS_R_32F 0 // float32
#define HIPBLAS_R_64F 1 // float64
#define HIPBLAS_R_16F 2 // float16
#define HIPBLAS_R_8I 4  // int8

typedef int hipblasComputeType_t;
typedef int hipblasGemmAlgo_t;
#define HIPBLAS_GEMM_DEFAULT 0

// Half type shim (layout compatible with __fp16 / _Float16)
typedef unsigned short hipblasHalf;

struct HipblasHandle
{
    hipStream_t stream;
};

extern "C"
{

    // ── Handle management ─────────────────────────────────────────────

    hipblasStatus_t hipblasCreate(void **handle)
    {
        if (!handle)
            return HIPBLAS_STATUS_INVALID_VALUE;
        *handle = new HipblasHandle{nullptr};
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t hipblasDestroy(void *handle)
    {
        delete reinterpret_cast<HipblasHandle *>(handle);
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t hipblasSetStream(void *handle, hipStream_t stream)
    {
        if (!handle)
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        reinterpret_cast<HipblasHandle *>(handle)->stream = stream;
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t hipblasGetStream(void *handle, hipStream_t *stream)
    {
        if (!handle || !stream)
            return HIPBLAS_STATUS_INVALID_VALUE;
        *stream = reinterpret_cast<HipblasHandle *>(handle)->stream;
        return HIPBLAS_STATUS_SUCCESS;
    }

    // ── SGEMM (FP32) ──────────────────────────────────────────────────
    // hipBLAS uses column-major; we flip A↔B and N↔M to convert to row-major
    hipblasStatus_t hipblasSgemm(
        void *handle,
        int transA, int transB,
        int m, int n, int k,
        const float *alpha,
        const float *A, int /*lda*/,
        const float *B, int /*ldb*/,
        const float *beta,
        float *C, int /*ldc*/)
    {
        if (!handle)
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        HipblasHandle *h = reinterpret_cast<HipblasHandle *>(handle);

        int ta = (transA == HIPBLAS_OP_T || transA == HIPBLAS_OP_C) ? 1 : 0;
        int tb = (transB == HIPBLAS_OP_T || transB == HIPBLAS_OP_C) ? 1 : 0;

        // Column-major C=A*B with dims (m,n,k) maps to row-major as:
        // row-major D = B^T * A^T with dims (n, m, k) — swap A/B and m/n
        vkflame_dispatch_linear(
            vkflame_get_context(),
            B, A, C, C, // swap A/B for col→row major conversion
            n, m, k,    // swap m/n
            0,          // DTYPE_FP32
            tb, ta,     // swap transflags too
            0,          // no activation
            alpha, beta, h->stream);

        return HIPBLAS_STATUS_SUCCESS;
    }

    // ── HGEMM (FP16) ──────────────────────────────────────────────────
    hipblasStatus_t hipblasHgemm(
        void *handle,
        int transA, int transB,
        int m, int n, int k,
        const hipblasHalf *alpha,
        const hipblasHalf *A, int /*lda*/,
        const hipblasHalf *B, int /*ldb*/,
        const hipblasHalf *beta,
        hipblasHalf *C, int /*ldc*/)
    {
        if (!handle)
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        HipblasHandle *h = reinterpret_cast<HipblasHandle *>(handle);

        int ta = (transA == HIPBLAS_OP_T || transA == HIPBLAS_OP_C) ? 1 : 0;
        int tb = (transB == HIPBLAS_OP_T || transB == HIPBLAS_OP_C) ? 1 : 0;

        vkflame_dispatch_linear(
            vkflame_get_context(),
            B, A, C, C,
            n, m, k,
            1, // DTYPE_FP16
            tb, ta, 0,
            alpha, beta, h->stream);

        return HIPBLAS_STATUS_SUCCESS;
    }

    // ── Batched SGEMM ─────────────────────────────────────────────────
    hipblasStatus_t hipblasSgemmBatched(
        void *handle,
        int transA, int transB,
        int m, int n, int k,
        const float *alpha,
        const float *const A_array[], int /*lda*/,
        const float *const B_array[], int /*ldb*/,
        const float *beta,
        float *const C_array[], int /*ldc*/,
        int batch_count)
    {
        if (!handle)
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        // Loop over batch — one dispatch per slice
        for (int b = 0; b < batch_count; b++)
        {
            hipblasSgemm(handle, transA, transB, m, n, k,
                         alpha, A_array[b], m, B_array[b], k,
                         beta, C_array[b], m);
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t hipblasHgemmBatched(
        void *handle,
        int transA, int transB,
        int m, int n, int k,
        const hipblasHalf *alpha,
        const hipblasHalf *const A_array[], int /*lda*/,
        const hipblasHalf *const B_array[], int /*ldb*/,
        const hipblasHalf *beta,
        hipblasHalf *const C_array[], int /*ldc*/,
        int batch_count)
    {
        if (!handle)
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        for (int b = 0; b < batch_count; b++)
        {
            hipblasHgemm(handle, transA, transB, m, n, k,
                         alpha, A_array[b], m, B_array[b], k,
                         beta, C_array[b], m);
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    // ── Strided batched SGEMM ─────────────────────────────────────────
    hipblasStatus_t hipblasSgemmStridedBatched(
        void *handle,
        int transA, int transB,
        int m, int n, int k,
        const float *alpha,
        const float *A, int /*lda*/, long long strideA,
        const float *B, int /*ldb*/, long long strideB,
        const float *beta,
        float *C, int /*ldc*/, long long strideC,
        int batch_count)
    {
        if (!handle)
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        HipblasHandle *h = reinterpret_cast<HipblasHandle *>(handle);
        int ta = (transA == HIPBLAS_OP_T || transA == HIPBLAS_OP_C) ? 1 : 0;
        int tb = (transB == HIPBLAS_OP_T || transB == HIPBLAS_OP_C) ? 1 : 0;
        for (int b = 0; b < batch_count; b++)
        {
            vkflame_dispatch_linear(
                vkflame_get_context(),
                B + b * strideB, A + b * strideA,
                C + b * strideC, C + b * strideC,
                n, m, k, 0, tb, ta, 0,
                alpha, beta, h->stream);
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t hipblasHgemmStridedBatched(
        void *handle,
        int transA, int transB,
        int m, int n, int k,
        const hipblasHalf *alpha,
        const hipblasHalf *A, int /*lda*/, long long strideA,
        const hipblasHalf *B, int /*ldb*/, long long strideB,
        const hipblasHalf *beta,
        hipblasHalf *C, int /*ldc*/, long long strideC,
        int batch_count)
    {
        if (!handle)
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        HipblasHandle *h = reinterpret_cast<HipblasHandle *>(handle);
        int ta = (transA == HIPBLAS_OP_T || transA == HIPBLAS_OP_C) ? 1 : 0;
        int tb = (transB == HIPBLAS_OP_T || transB == HIPBLAS_OP_C) ? 1 : 0;
        for (int b = 0; b < batch_count; b++)
        {
            vkflame_dispatch_linear(
                vkflame_get_context(),
                B + b * strideB, A + b * strideA,
                C + b * strideC, C + b * strideC,
                n, m, k, 1, tb, ta, 0,
                alpha, beta, h->stream);
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    // ── hipblasGemmEx — explicit-type GEMM (columns-major, swap A↔B/m↔n) ──
    // ggml calls this with FP16 A/B and FP32 C output (HIPBLAS_COMPUTE_32F).
    // We detect that pattern and route to VKF_DTYPE_FP16_FP32OUT so the kernel
    // writes float32, not float16, into the output buffer.
    hipblasStatus_t hipblasGemmEx(
        void *handle,
        int transA, int transB,
        int m, int n, int k,
        const void *alpha,
        const void *A, hipblasDatatype_t aType, int /*lda*/,
        const void *B, hipblasDatatype_t bType, int /*ldb*/,
        const void *beta,
        void *C, hipblasDatatype_t cType, int /*ldc*/,
        hipblasComputeType_t /*computeType*/, hipblasGemmAlgo_t /*algo*/)
    {
        if (!handle)
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        HipblasHandle *h = reinterpret_cast<HipblasHandle *>(handle);

        int ta = (transA == HIPBLAS_OP_T || transA == HIPBLAS_OP_C) ? 1 : 0;
        int tb = (transB == HIPBLAS_OP_T || transB == HIPBLAS_OP_C) ? 1 : 0;

        // Map type pair → our dispatch dtype
        int dtype;
        if (aType == HIPBLAS_R_16F && cType == HIPBLAS_R_32F)
            dtype = VKF_DTYPE_FP16_FP32OUT; // fp16 inputs, fp32 output
        else if (aType == HIPBLAS_R_16F)
            dtype = VKF_DTYPE_FP16;
        else if (aType == HIPBLAS_R_8I)
            dtype = VKF_DTYPE_INT8;
        else
            dtype = VKF_DTYPE_FP32;

        (void)bType; // B type follows A type in our kernels

        vkflame_dispatch_linear(
            vkflame_get_context(),
            B, A, C, C,    // column-major: swap A↔B
            n, m, k,       // swap m↔n
            dtype, tb, ta, // swap trans flags too
            0, alpha, beta, h->stream);

        return HIPBLAS_STATUS_SUCCESS;
    }

    // ── hipblasGemmBatchedEx ───────────────────────────────────────────
    hipblasStatus_t hipblasGemmBatchedEx(
        void *handle,
        int transA, int transB,
        int m, int n, int k,
        const void *alpha,
        const void *const A_array[], hipblasDatatype_t aType, int /*lda*/,
        const void *const B_array[], hipblasDatatype_t bType, int /*ldb*/,
        const void *beta,
        void *const C_array[], hipblasDatatype_t cType, int /*ldc*/,
        int batch_count,
        hipblasComputeType_t computeType, hipblasGemmAlgo_t algo)
    {
        if (!handle)
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        for (int b = 0; b < batch_count; b++)
            hipblasGemmEx(handle, transA, transB, m, n, k,
                          alpha, A_array[b], aType, m,
                          B_array[b], bType, k,
                          beta, C_array[b], cType, m,
                          computeType, algo);
        return HIPBLAS_STATUS_SUCCESS;
    }

    // ── hipblasGemmStridedBatchedEx ────────────────────────────────────
    hipblasStatus_t hipblasGemmStridedBatchedEx(
        void *handle,
        int transA, int transB,
        int m, int n, int k,
        const void *alpha,
        const void *A, hipblasDatatype_t aType, int /*lda*/, long long strideA,
        const void *B, hipblasDatatype_t bType, int /*ldb*/, long long strideB,
        const void *beta,
        void *C, hipblasDatatype_t cType, int /*ldc*/, long long strideC,
        int batch_count,
        hipblasComputeType_t computeType, hipblasGemmAlgo_t algo)
    {
        if (!handle)
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        // Compute element size for pointer arithmetic
        int elem_a = (aType == HIPBLAS_R_16F) ? 2 : 4;
        int elem_b = (bType == HIPBLAS_R_16F) ? 2 : 4;
        int elem_c = (cType == HIPBLAS_R_32F) ? 4 : 2;
        const uint8_t *pA = reinterpret_cast<const uint8_t *>(A);
        const uint8_t *pB = reinterpret_cast<const uint8_t *>(B);
        uint8_t *pC = reinterpret_cast<uint8_t *>(C);
        for (int b = 0; b < batch_count; b++)
        {
            hipblasGemmEx(handle, transA, transB, m, n, k,
                          alpha,
                          pA + (long long)b * strideA * elem_a, aType, m,
                          pB + (long long)b * strideB * elem_b, bType, k,
                          beta,
                          pC + (long long)b * strideC * elem_c, cType, m,
                          computeType, algo);
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    // ── hipblasStrsmBatched — triangular solve stub ────────────────────
    // Not used for LLM inference; return not-supported so calling code falls
    // back gracefully (ggml only calls this for some CUDA-specific paths).
    hipblasStatus_t hipblasStrsmBatched(
        void * /*handle*/, int /*side*/, int /*uplo*/, int /*transA*/, int /*diag*/,
        int /*m*/, int /*n*/,
        const float * /*alpha*/,
        const float *const /*A*/[], int /*lda*/,
        float *const /*B*/[], int /*ldb*/,
        int /*batch_count*/)
    {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

} // extern "C"
