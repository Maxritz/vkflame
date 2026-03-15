// hipblaslt_shim.cpp — vkflame drop-in for hipBLASLt API
// Routes hipblasLtMatmul to vkflame_dispatch_linear.

#include "../runtime/device.h"
#include "../runtime/dispatch.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ── hipBLASLt type definitions ────────────────────────────────────
typedef int hipblasStatus_t;
typedef void *hipStream_t;

#define HIPBLAS_STATUS_SUCCESS 0
#define HIPBLAS_STATUS_INVALID_VALUE 3
#define HIPBLAS_STATUS_NOT_SUPPORTED 8

// Attribute enums (partial — just what inference code uses)
#define HIPBLASLT_MATRIX_LAYOUT_ROWS 0
#define HIPBLASLT_MATRIX_LAYOUT_COLS 1
#define HIPBLASLT_MATRIX_LAYOUT_LD 2
#define HIPBLASLT_MATRIX_LAYOUT_TYPE 3
#define HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT 4
#define HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET 5

#define HIPBLASLT_MATMUL_DESC_TRANSA 0
#define HIPBLASLT_MATMUL_DESC_TRANSB 1
#define HIPBLASLT_MATMUL_DESC_COMPUTE_TYPE 2
#define HIPBLASLT_MATMUL_DESC_SCALE_TYPE 3
#define HIPBLASLT_MATMUL_DESC_POINTER_MODE 4
#define HIPBLASLT_MATMUL_DESC_EPILOGUE 5
#define HIPBLASLT_MATMUL_DESC_BIAS_POINTER 6
#define HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER 7
#define HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER 8
#define HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER 9
#define HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER 10

// EPILOGUE flags
#define HIPBLASLT_EPILOGUE_DEFAULT 0
#define HIPBLASLT_EPILOGUE_RELU 1
#define HIPBLASLT_EPILOGUE_BIAS 2
#define HIPBLASLT_EPILOGUE_GELU 3
#define HIPBLASLT_EPILOGUE_SILU 4

// Data types
#define HIPBLASLT_R_16F 0 // FP16
#define HIPBLASLT_R_32F 1 // FP32
#define HIPBLASLT_R_8I 2  // INT8
#define HIPBLASLT_R_16B 3 // BF16

// Op N/T
#define HIPBLAS_OP_N 111
#define HIPBLAS_OP_T 112
#define HIPBLAS_OP_C 113

// Preference attribute enums
#define HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES 0

// ── Handle structs ────────────────────────────────────────────────

struct HipblasLtHandle
{
    hipStream_t stream;
};

struct MatrixLayout
{
    uint64_t rows;
    uint64_t cols;
    uint64_t ld;
    int dtype; // HIPBLASLT_R_*
    int batch_count;
    int64_t stride;
};

struct MatmulDesc
{
    int transA; // HIPBLAS_OP_N or _T
    int transB;
    int compute_type;
    int epilogue; // HIPBLASLT_EPILOGUE_*
    void *bias_ptr;
    void *a_scale_ptr;
    void *b_scale_ptr;
};

struct MatmulPreference
{
    uint64_t max_workspace_bytes;
};

// One heuristic entry (opaque to caller)
struct HeuristicResult
{
    int algo_id;
};

extern "C"
{

    // ── Handle management ─────────────────────────────────────────────

    hipblasStatus_t hipblasLtCreate(void **lightHandle)
    {
        if (!lightHandle)
            return HIPBLAS_STATUS_INVALID_VALUE;
        *lightHandle = new HipblasLtHandle{nullptr};
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t hipblasLtDestroy(void *lightHandle)
    {
        delete reinterpret_cast<HipblasLtHandle *>(lightHandle);
        return HIPBLAS_STATUS_SUCCESS;
    }

    // ── Matrix layout ─────────────────────────────────────────────────

    hipblasStatus_t hipblasLtMatrixLayoutCreate(
        void **matLayout, int type, uint64_t rows, uint64_t cols, int64_t /*ld*/)
    {
        if (!matLayout)
            return HIPBLAS_STATUS_INVALID_VALUE;
        auto *m = new MatrixLayout{};
        m->rows = rows;
        m->cols = cols;
        m->ld = cols; // row-major default
        m->dtype = type;
        m->batch_count = 1;
        m->stride = 0;
        *matLayout = m;
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t hipblasLtMatrixLayoutDestroy(void *matLayout)
    {
        delete reinterpret_cast<MatrixLayout *>(matLayout);
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t hipblasLtMatrixLayoutSetAttribute(
        void *matLayout, int attr, const void *buf, size_t sizeInBytes)
    {
        (void)sizeInBytes;
        if (!matLayout || !buf)
            return HIPBLAS_STATUS_INVALID_VALUE;
        MatrixLayout *m = reinterpret_cast<MatrixLayout *>(matLayout);
        switch (attr)
        {
        case HIPBLASLT_MATRIX_LAYOUT_ROWS:
            m->rows = *reinterpret_cast<const uint64_t *>(buf);
            break;
        case HIPBLASLT_MATRIX_LAYOUT_COLS:
            m->cols = *reinterpret_cast<const uint64_t *>(buf);
            break;
        case HIPBLASLT_MATRIX_LAYOUT_LD:
            m->ld = *reinterpret_cast<const uint64_t *>(buf);
            break;
        case HIPBLASLT_MATRIX_LAYOUT_TYPE:
            m->dtype = *reinterpret_cast<const int *>(buf);
            break;
        case HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
            m->batch_count = *reinterpret_cast<const int *>(buf);
            break;
        case HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
            m->stride = *reinterpret_cast<const int64_t *>(buf);
            break;
        default:
            break;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t hipblasLtMatrixLayoutGetAttribute(
        void *matLayout, int attr, void *buf, size_t sizeInBytes, size_t *sizeWritten)
    {
        (void)sizeInBytes;
        if (!matLayout || !buf)
            return HIPBLAS_STATUS_INVALID_VALUE;
        MatrixLayout *m = reinterpret_cast<MatrixLayout *>(matLayout);
        if (sizeWritten)
            *sizeWritten = 8;
        switch (attr)
        {
        case HIPBLASLT_MATRIX_LAYOUT_ROWS:
            *reinterpret_cast<uint64_t *>(buf) = m->rows;
            break;
        case HIPBLASLT_MATRIX_LAYOUT_COLS:
            *reinterpret_cast<uint64_t *>(buf) = m->cols;
            break;
        case HIPBLASLT_MATRIX_LAYOUT_LD:
            *reinterpret_cast<uint64_t *>(buf) = m->ld;
            break;
        case HIPBLASLT_MATRIX_LAYOUT_TYPE:
            *reinterpret_cast<int *>(buf) = m->dtype;
            if (sizeWritten)
                *sizeWritten = 4;
            break;
        case HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
            *reinterpret_cast<int *>(buf) = m->batch_count;
            if (sizeWritten)
                *sizeWritten = 4;
            break;
        default:
            break;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    // ── Matmul descriptor ─────────────────────────────────────────────

    hipblasStatus_t hipblasLtMatmulDescCreate(
        void **matmulDesc, int computeType, int scaleType)
    {
        (void)scaleType;
        if (!matmulDesc)
            return HIPBLAS_STATUS_INVALID_VALUE;
        auto *d = new MatmulDesc{};
        d->transA = HIPBLAS_OP_N;
        d->transB = HIPBLAS_OP_N;
        d->compute_type = computeType;
        d->epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
        d->bias_ptr = nullptr;
        d->a_scale_ptr = nullptr;
        d->b_scale_ptr = nullptr;
        *matmulDesc = d;
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t hipblasLtMatmulDescDestroy(void *matmulDesc)
    {
        delete reinterpret_cast<MatmulDesc *>(matmulDesc);
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t hipblasLtMatmulDescSetAttribute(
        void *matmulDesc, int attr, const void *buf, size_t /*sizeInBytes*/)
    {
        if (!matmulDesc || !buf)
            return HIPBLAS_STATUS_INVALID_VALUE;
        MatmulDesc *d = reinterpret_cast<MatmulDesc *>(matmulDesc);
        switch (attr)
        {
        case HIPBLASLT_MATMUL_DESC_TRANSA:
            d->transA = *reinterpret_cast<const int *>(buf);
            break;
        case HIPBLASLT_MATMUL_DESC_TRANSB:
            d->transB = *reinterpret_cast<const int *>(buf);
            break;
        case HIPBLASLT_MATMUL_DESC_EPILOGUE:
            d->epilogue = *reinterpret_cast<const int *>(buf);
            break;
        case HIPBLASLT_MATMUL_DESC_BIAS_POINTER:
            d->bias_ptr = *reinterpret_cast<void **>(const_cast<void *>(buf));
            break;
        case HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER:
            d->a_scale_ptr = *reinterpret_cast<void **>(const_cast<void *>(buf));
            break;
        case HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER:
            d->b_scale_ptr = *reinterpret_cast<void **>(const_cast<void *>(buf));
            break;
        default:
            break;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t hipblasLtMatmulDescGetAttribute(
        void *matmulDesc, int attr, void *buf, size_t /*sizeInBytes*/, size_t *sizeWritten)
    {
        if (!matmulDesc || !buf)
            return HIPBLAS_STATUS_INVALID_VALUE;
        MatmulDesc *d = reinterpret_cast<MatmulDesc *>(matmulDesc);
        if (sizeWritten)
            *sizeWritten = 4;
        switch (attr)
        {
        case HIPBLASLT_MATMUL_DESC_TRANSA:
            *reinterpret_cast<int *>(buf) = d->transA;
            break;
        case HIPBLASLT_MATMUL_DESC_TRANSB:
            *reinterpret_cast<int *>(buf) = d->transB;
            break;
        case HIPBLASLT_MATMUL_DESC_EPILOGUE:
            *reinterpret_cast<int *>(buf) = d->epilogue;
            break;
        default:
            break;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    // ── Preference ────────────────────────────────────────────────────

    hipblasStatus_t hipblasLtMatmulPreferenceCreate(void **pref)
    {
        if (!pref)
            return HIPBLAS_STATUS_INVALID_VALUE;
        *pref = new MatmulPreference{};
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t hipblasLtMatmulPreferenceDestroy(void *pref)
    {
        delete reinterpret_cast<MatmulPreference *>(pref);
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(
        void *pref, int attr, const void *buf, size_t /*sizeInBytes*/)
    {
        if (!pref || !buf)
            return HIPBLAS_STATUS_INVALID_VALUE;
        MatmulPreference *p = reinterpret_cast<MatmulPreference *>(pref);
        if (attr == HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES)
            p->max_workspace_bytes = *reinterpret_cast<const uint64_t *>(buf);
        return HIPBLAS_STATUS_SUCCESS;
    }

    // ── Heuristic ─────────────────────────────────────────────────────

    hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(
        void * /*lightHandle*/,
        void * /*matmulDesc*/,
        void * /*Adesc*/,
        void * /*Bdesc*/,
        void * /*Cdesc*/,
        void * /*Ddesc*/,
        void * /*pref*/,
        int requestedAlgoCount,
        void *heuristicResultsArray,
        int *returnAlgoCount)
    {
        // Return exactly one trivial algorithm
        if (returnAlgoCount)
            *returnAlgoCount = (requestedAlgoCount > 0 ? 1 : 0);
        if (heuristicResultsArray && requestedAlgoCount > 0)
        {
            reinterpret_cast<HeuristicResult *>(heuristicResultsArray)->algo_id = 0;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    // ── Matmul — the only function with real logic ────────────────────

    hipblasStatus_t hipblasLtMatmul(
        void *lightHandle,
        void *matmulDesc,
        const void *alpha,
        const void *A, void *Adesc,
        const void *B, void *Bdesc,
        const void *beta,
        const void *C, void *Cdesc,
        void *D, void *Ddesc,
        const void * /*algo*/,
        void * /*workspace*/,
        size_t /*workspaceSizeInBytes*/,
        hipStream_t stream)
    {
        if (!lightHandle || !matmulDesc)
            return HIPBLAS_STATUS_INVALID_VALUE;

        MatmulDesc *desc = reinterpret_cast<MatmulDesc *>(matmulDesc);
        MatrixLayout *Alay = reinterpret_cast<MatrixLayout *>(Adesc);
        MatrixLayout *Blay = reinterpret_cast<MatrixLayout *>(Bdesc);
        MatrixLayout *Dlay = reinterpret_cast<MatrixLayout *>(Ddesc);

        // Extract M, N, K from layout descriptors
        // Convention: A is M×K, B is K×N, D is M×N (row-major)
        int M = (int)(Alay ? Alay->rows : Dlay->rows);
        int N = (int)(Blay ? Blay->cols : Dlay->cols);
        int K = (int)(Alay ? Alay->cols : (Blay ? Blay->rows : 0));

        // Extract dtype from A layout
        int dtype = 0; // FP32 default
        if (Alay)
        {
            switch (Alay->dtype)
            {
            case HIPBLASLT_R_16F:
                dtype = 1;
                break; // FP16
            case HIPBLASLT_R_32F:
                dtype = 0;
                break; // FP32
            case HIPBLASLT_R_8I:
                dtype = 2;
                break; // INT8
            case HIPBLASLT_R_16B:
                dtype = 3;
                break; // BF16
            }
        }

        // Map trans flags
        int transA = (desc->transA == HIPBLAS_OP_T || desc->transA == HIPBLAS_OP_C) ? 1 : 0;
        int transB = (desc->transB == HIPBLAS_OP_T || desc->transB == HIPBLAS_OP_C) ? 1 : 0;

        // Map epilogue → activation: RELU→2, GELU→3, SILU→1, NONE/DEFAULT→0
        int activation = 0;
        switch (desc->epilogue)
        {
        case HIPBLASLT_EPILOGUE_RELU:
            activation = 2;
            break;
        case HIPBLASLT_EPILOGUE_GELU:
            activation = 3;
            break;
        case HIPBLASLT_EPILOGUE_SILU:
            activation = 1;
            break;
        default:
            activation = 0;
            break;
        }

        vkflame_dispatch_linear(
            vkflame_get_context(),
            A, B, C, D,
            M, N, K,
            dtype, transA, transB, activation,
            alpha, beta, stream);

        return HIPBLAS_STATUS_SUCCESS;
    }

} // extern "C"
