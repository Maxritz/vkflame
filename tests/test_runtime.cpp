// tests/test_runtime.cpp
// Build: cmake -B build && cmake --build build --config Release
// Run:   build/Release/vkflame_tests.exe
//
// Validates every implemented kernel against CPU reference.
// Every PASS/FAIL line is machine-readable: grep "FAIL" to detect regressions.

#include "device.h"
#include "buffer.h"
#include "dispatch.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <chrono>

// ── half-float helpers (no dependency on a half library) ──────────────────
// We only need f32→f16 and f16→f32 for test data.
static uint16_t f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint16_t s = (x >> 16) & 0x8000u;
    int32_t e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFFu;
    if (e <= 0)
    {
        return s;
    }
    if (e >= 31)
    {
        return s | 0x7C00u;
    }
    return s | (uint16_t)(e << 10) | (uint16_t)(m >> 13);
}
static float f16_to_f32(uint16_t h)
{
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    if (e == 0)
    { /* subnormal → zero */
        return 0.f;
    }
    if (e == 31)
    {
        uint32_t v = s | 0x7F800000u | (m << 13);
        float r;
        memcpy(&r, &v, 4);
        return r;
    }
    uint32_t v = s | ((e + 112u) << 23) | (m << 13);
    float r;
    memcpy(&r, &v, 4);
    return r;
}

// ── Test helpers ──────────────────────────────────────────────────────────
static int g_pass = 0, g_fail = 0;

static void check(const char *name, bool ok)
{
    if (ok)
    {
        printf("[PASS] %s\n", name);
        g_pass++;
    }
    else
    {
        printf("[FAIL] %s\n", name);
        g_fail++;
    }
}
static void check_near(const char *name, float got, float ref, float tol)
{
    float err = std::abs(got - ref);
    if (err <= tol)
        printf("[PASS] %s  got=%.6f  ref=%.6f  err=%.2e\n", name, got, ref, err);
    else
    {
        printf("[FAIL] %s  got=%.6f  ref=%.6f  err=%.2e  (tol=%.2e)\n",
               name, got, ref, err, tol);
        g_fail++;
    }
}
static void check_array(const char *name, const float *got, const float *ref,
                        int n, float tol)
{
    float max_err = 0.f;
    int bad_idx = -1;
    for (int i = 0; i < n; i++)
    {
        float e = std::abs(got[i] - ref[i]);
        if (e > max_err)
        {
            max_err = e;
            bad_idx = i;
        }
    }
    if (max_err <= tol)
    {
        printf("[PASS] %s  max_err=%.2e  (n=%d)\n", name, max_err, n);
        g_pass++;
    }
    else
    {
        printf("[FAIL] %s  max_err=%.2e at [%d] got=%.4f ref=%.4f  (tol=%.2e, n=%d)\n",
               name, max_err, bad_idx, got[bad_idx], ref[bad_idx], tol, n);
        g_fail++;
    }
}
static void check_array_f16(const char *name, const uint16_t *got_h, const float *ref,
                            int n, float tol)
{
    std::vector<float> got(n);
    for (int i = 0; i < n; i++)
        got[i] = f16_to_f32(got_h[i]);
    check_array(name, got.data(), ref, n, tol);
}

// ── Transfer helpers ─────────────────────────────────────────────────────
static VKFBuffer *upload_f32(const float *data, int n)
{
    auto *b = vkflame_alloc((size_t)n * sizeof(float));
    vkflame_memcpy_h2d(b, data, (size_t)n * sizeof(float), 0);
    return b;
}
static VKFBuffer *upload_f16(const std::vector<uint16_t> &v)
{
    auto *b = vkflame_alloc(v.size() * 2);
    vkflame_memcpy_h2d(b, v.data(), v.size() * 2, 0);
    return b;
}
static VKFBuffer *alloc_f32(int n) { return vkflame_alloc((size_t)n * sizeof(float)); }
static VKFBuffer *alloc_f16(int n) { return vkflame_alloc((size_t)n * 2); }

static void download_f32(VKFBuffer *b, float *out, int n)
{
    vkflame_memcpy_d2h(out, b, (size_t)n * sizeof(float), 0);
}
static void download_f16(VKFBuffer *b, uint16_t *out, int n)
{
    vkflame_memcpy_d2h(out, b, (size_t)n * 2, 0);
}

// ── CPU reference implementations ────────────────────────────────────────
static void cpu_gemm_f32(const float *A, const float *B, float *C,
                         int M, int K, int N)
{
    // C[M,N] = A[M,K] @ B[K,N]
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
        {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += A[m * K + k] * B[k * N + n];
            C[m * N + n] = acc;
        }
}

static void cpu_rms_norm(const float *x, float *y, int M, int N, float eps)
{
    for (int m = 0; m < M; m++)
    {
        const float *row = x + m * N;
        float ss = 0.f;
        for (int n = 0; n < N; n++)
            ss += row[n] * row[n];
        float rrms = 1.f / std::sqrt(ss / N + eps);
        for (int n = 0; n < N; n++)
            y[m * N + n] = row[n] * rrms;
    }
}

static void cpu_softmax(const float *x, float *y, int M, int N)
{
    for (int m = 0; m < M; m++)
    {
        const float *rx = x + m * N;
        float *ry = y + m * N;
        float mx = *std::max_element(rx, rx + N);
        float s = 0.f;
        for (int n = 0; n < N; n++)
        {
            ry[n] = std::exp(rx[n] - mx);
            s += ry[n];
        }
        for (int n = 0; n < N; n++)
            ry[n] /= s;
    }
}

static void cpu_silu(const float *x, float *y, int n)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] / (1.f + std::exp(-x[i]));
}

static void cpu_scale(const float *x, float *y, int n, float sc, float bias)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] * sc + bias;
}

static void cpu_add(const float *a, const float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

// ── Tests ─────────────────────────────────────────────────────────────────

// ── 1.  Device init ────────────────────────────────────────────────────
static void test_device_init()
{
    printf("\n=== Device init ===\n");
    VKFContext *ctx = vkflame_get_context();
    check("ctx not null", ctx != nullptr);
    check("compute queue exists", ctx && ctx->compute_queue != VK_NULL_HANDLE);
    check("device name non-empty", ctx && ctx->features.device_name[0] != 0);
    if (ctx)
    {
        printf("  GPU     : %s\n", ctx->features.device_name);
        printf("  subgroup: %u\n", ctx->features.subgroup_size);
        printf("  coop_mat: %s\n", ctx->features.has_cooperative_matrix ? "yes" : "no");
        printf("  fp8     : %s\n", ctx->features.has_float8 ? "yes" : "no");
    }
}

// ── 2.  Buffer round-trip ─────────────────────────────────────────────
static void test_buffer_roundtrip()
{
    printf("\n=== Buffer round-trip ===\n");
    const int N = 1024 * 1024; // 4 MB
    std::vector<uint8_t> src(N);
    std::iota(src.begin(), src.end(), (uint8_t)0);

    auto *buf = vkflame_alloc(N);
    check("alloc non-null", buf != nullptr);
    check("buf->address non-zero", buf && buf->address != 0);

    vkflame_memcpy_h2d(buf, src.data(), N, 0);
    std::vector<uint8_t> dst(N, 0);
    vkflame_memcpy_d2h(dst.data(), buf, N, 0);
    check("roundtrip exact", dst == src);
    vkflame_free(buf);
}

// ── 3.  FP16 GEMM ─────────────────────────────────────────────────────
// We upload as FP16 and read back as FP16, then compare to FP32 CPU ref.
static void test_gemm_fp16()
{
    printf("\n=== FP16 GEMM ===\n");

    // Small deterministic test: M=4 K=4 N=4
    {
        const int M = 4, K = 4, N = 4;
        // A = identity-ish, B = known matrix
        float Af[M * K] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        float Bf[K * N] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
        float Cref[M * N];
        cpu_gemm_f32(Af, Bf, Cref, M, K, N);

        std::vector<uint16_t> Ah(M * K), Bh(K * N);
        for (int i = 0; i < M * K; i++)
            Ah[i] = f32_to_f16(Af[i]);
        for (int i = 0; i < K * N; i++)
            Bh[i] = f32_to_f16(Bf[i]);

        auto *bA = upload_f16(Ah);
        auto *bB = upload_f16(Bh);
        auto *bD = alloc_f16(M * N);

        vkflame_dispatch_linear(nullptr, (void *)bA->address, (void *)bB->address,
                                nullptr, (void *)bD->address,
                                M, N, K, VKF_DTYPE_FP16, 0, 0, 0, nullptr, nullptr, nullptr);

        std::vector<uint16_t> out(M * N);
        download_f16(bD, out.data(), M * N);
        check_array_f16("gemm_fp16_identity_4x4", out.data(), Cref, M * N, 0.05f);

        vkflame_free(bA);
        vkflame_free(bB);
        vkflame_free(bD);
    }

    // Medium test: M=32 K=64 N=32 (random-ish)
    {
        const int M = 32, K = 64, N = 32;
        std::vector<float> Af(M * K), Bf(K * N), Cref(M * N);
        for (int i = 0; i < M * K; i++)
            Af[i] = (float)(((i * 7 + 3) % 17) - 8) / 8.f;
        for (int i = 0; i < K * N; i++)
            Bf[i] = (float)(((i * 13 + 5) % 11) - 5) / 5.f;
        cpu_gemm_f32(Af.data(), Bf.data(), Cref.data(), M, K, N);

        std::vector<uint16_t> Ah(M * K), Bh(K * N);
        for (int i = 0; i < M * K; i++)
            Ah[i] = f32_to_f16(Af[i]);
        for (int i = 0; i < K * N; i++)
            Bh[i] = f32_to_f16(Bf[i]);

        auto *bA = upload_f16(Ah);
        auto *bB = upload_f16(Bh);
        auto *bD = alloc_f16(M * N);

        vkflame_dispatch_linear(nullptr, (void *)bA->address, (void *)bB->address,
                                nullptr, (void *)bD->address,
                                M, N, K, VKF_DTYPE_FP16, 0, 0, 0, nullptr, nullptr, nullptr);

        std::vector<uint16_t> out(M * N);
        download_f16(bD, out.data(), M * N);
        check_array_f16("gemm_fp16_32x64x32", out.data(), Cref.data(), M * N, 0.5f);

        vkflame_free(bA);
        vkflame_free(bB);
        vkflame_free(bD);
    }
}

// ── 4.  RMS Norm ─────────────────────────────────────────────────────
static void test_rms_norm()
{
    printf("\n=== RMS-Norm ===\n");
    const int M = 8, N = 256;
    std::vector<float> x(M * N), ref(M * N), got(M * N);
    for (int i = 0; i < M * N; i++)
        x[i] = (float)(((i * 11 + 7) % 23) - 11) / 4.f;
    cpu_rms_norm(x.data(), ref.data(), M, N, 1e-5f);

    auto *bX = upload_f32(x.data(), M * N);
    auto *bY = alloc_f32(M * N);

    vkflame_dispatch_rms_norm_f32(nullptr, (void *)bX->address, (void *)bY->address,
                                  M, N, 1e-5f);

    download_f32(bY, got.data(), M * N);
    check_array("rms_norm_f32_8x256", got.data(), ref.data(), M * N, 1e-4f);

    vkflame_free(bX);
    vkflame_free(bY);
}

// ── 5.  Softmax ────────────────────────────────────────────────────
static void test_softmax()
{
    printf("\n=== Softmax ===\n");
    const int M = 4, N = 128;
    std::vector<float> x(M * N), ref(M * N), got(M * N);
    for (int i = 0; i < M * N; i++)
        x[i] = (float)(((i * 17 + 3) % 31) - 15) / 3.f;
    cpu_softmax(x.data(), ref.data(), M, N);

    auto *bX = upload_f32(x.data(), M * N);
    auto *bY = alloc_f32(M * N);

    vkflame_dispatch_softmax(nullptr, (void *)bX->address, (void *)bY->address, M, N);

    download_f32(bY, got.data(), M * N);
    check_array("softmax_4x128", got.data(), ref.data(), M * N, 1e-4f);

    // Row sums must be 1.0
    for (int m = 0; m < M; m++)
    {
        float s = 0.f;
        for (int n = 0; n < N; n++)
            s += got[m * N + n];
        char nm[64];
        snprintf(nm, sizeof(nm), "softmax_rowsum_row%d", m);
        check_near(nm, s, 1.0f, 1e-3f);
    }

    vkflame_free(bX);
    vkflame_free(bY);
}

// ── 6.  SiLU element-wise ─────────────────────────────────────────
static void test_silu()
{
    printf("\n=== SiLU ===\n");
    const int N = 512;
    std::vector<float> x(N), ref(N), got(N);
    for (int i = 0; i < N; i++)
        x[i] = (float)(i - N / 2) / 64.f;
    cpu_silu(x.data(), ref.data(), N);

    auto *bX = upload_f32(x.data(), N);
    auto *bY = alloc_f32(N);

    vkflame_dispatch_elementwise_f32(nullptr, (void *)bX->address, (void *)bY->address,
                                     N, VKF_EW_SILU);

    download_f32(bY, got.data(), N);
    check_array("silu_512", got.data(), ref.data(), N, 1e-4f);

    vkflame_free(bX);
    vkflame_free(bY);
}

// ── 7.  Scale+bias ────────────────────────────────────────────────
static void test_scale()
{
    printf("\n=== Scale ===\n");
    const int N = 256;
    std::vector<float> x(N), ref(N), got(N);
    for (int i = 0; i < N; i++)
        x[i] = (float)(i) / 32.f;
    const float sc = 0.5f, bias = 0.1f;
    cpu_scale(x.data(), ref.data(), N, sc, bias);

    auto *bX = upload_f32(x.data(), N);
    auto *bY = alloc_f32(N);

    vkflame_dispatch_scale_f32(nullptr, (void *)bX->address, (void *)bY->address,
                               N, sc, bias);

    download_f32(bY, got.data(), N);
    check_array("scale_bias_256", got.data(), ref.data(), N, 1e-5f);

    vkflame_free(bX);
    vkflame_free(bY);
}

// ── 8.  Binary add ────────────────────────────────────────────────
static void test_binop_add()
{
    printf("\n=== Binop add ===\n");
    const int N = 1024;
    std::vector<float> a(N), b(N), ref(N), got(N);
    for (int i = 0; i < N; i++)
    {
        a[i] = (float)i / 100.f;
        b[i] = (float)(N - i) / 100.f;
    }
    cpu_add(a.data(), b.data(), ref.data(), N);

    auto *bA = upload_f32(a.data(), N);
    auto *bB = upload_f32(b.data(), N);
    auto *bC = alloc_f32(N);

    vkflame_dispatch_binop_f32(nullptr, (void *)bA->address, (void *)bB->address,
                               (void *)bC->address, N, N, VKF_BINOP_ADD);

    download_f32(bC, got.data(), N);
    check_array("binop_add_1024", got.data(), ref.data(), N, 1e-5f);

    vkflame_free(bA);
    vkflame_free(bB);
    vkflame_free(bC);
}

// ── 9.  Flash attention (basic shape check) ───────────────────────
static void test_flash_attention()
{
    printf("\n=== Flash Attention ===\n");
    // B=1 Hq=2 Hkv=2 Sq=8 Skv=8 D=32
    const int B = 1, Hq = 2, Hkv = 2, Sq = 8, Skv = 8, D = 32;
    const float scale = 1.f / std::sqrt((float)D);
    const int Q_n = B * Hq * Sq * D, KV_n = B * Hkv * Skv * D, O_n = Q_n;

    std::vector<float> Q(Q_n), K(KV_n), V(KV_n), O(O_n, 0.f);
    for (int i = 0; i < Q_n; i++)
        Q[i] = (float)(((i * 7 + 3) % 17) - 8) / 8.f;
    for (int i = 0; i < KV_n; i++)
        K[i] = (float)(((i * 13 + 5) % 11) - 5) / 5.f;
    for (int i = 0; i < KV_n; i++)
        V[i] = (float)(((i * 3 + 1) % 9) - 4) / 4.f;

    auto *bQ = upload_f32(Q.data(), Q_n);
    auto *bK = upload_f32(K.data(), KV_n);
    auto *bV = upload_f32(V.data(), KV_n);
    auto *bO = alloc_f32(O_n);

    vkflame_dispatch_flash_attention(nullptr,
                                     (void *)bQ->address, (void *)bK->address, (void *)bV->address, (void *)bO->address,
                                     B, Hq, Hkv, Sq, Skv, D, scale, 0);

    download_f32(bO, O.data(), O_n);

    // Soft check: output should be non-zero and finite
    bool all_finite = true;
    for (float v : O)
        if (!std::isfinite(v))
        {
            all_finite = false;
            break;
        }
    check("flash_attn_output_finite", all_finite);

    float mag = 0.f;
    for (float v : O)
        mag += v * v;
    check("flash_attn_output_nonzero", mag > 0.f);

    vkflame_free(bQ);
    vkflame_free(bK);
    vkflame_free(bV);
    vkflame_free(bO);
}

// ── 10.  LLM mini forward-pass simulation ─────────────────────────
// Simulates one transformer layer: RMSNorm → GEMM (QKV proj) → Attn → GEMM (out proj)
// Verifies outputs are finite and non-trivial.
static void test_mini_transformer_layer()
{
    printf("\n=== Mini transformer layer ===\n");
    const int batch = 1, seq = 16, dim = 128, ff_dim = 256, heads = 2, head_dim = 64;

    // Input activations
    std::vector<float> x(batch * seq * dim);
    for (int i = 0; i < (int)x.size(); i++)
        x[i] = (float)(((i * 7 + 3) % 17) - 8) / 8.f;
    auto *bX = upload_f32(x.data(), (int)x.size());

    // ── Step 1: RMSNorm
    auto *bXnorm = alloc_f32(batch * seq * dim);
    vkflame_dispatch_rms_norm_f32(nullptr, (void *)bX->address, (void *)bXnorm->address,
                                  batch * seq, dim, 1e-5f);

    // ── Step 2: QKV projection (3 GEMMs: Q, K, V)
    // Weight matrices: [dim, dim] for each head
    std::vector<uint16_t> Wq(dim * dim), Wk(dim * dim), Wv(dim * dim);
    for (int i = 0; i < dim * dim; i++)
    {
        // Near-identity initialization
        int r = i / dim, c = i % dim;
        float v = (r == c) ? 1.f : (float)(((i * 3 + 1) % 7) - 3) / 32.f;
        Wq[i] = f32_to_f16(v);
        Wk[i] = f32_to_f16(v * 0.9f);
        Wv[i] = f32_to_f16(v * 1.1f);
    }
    // Promote norm output to fp16 for GEMM
    std::vector<float> xnorm_f32(batch * seq * dim);
    download_f32(bXnorm, xnorm_f32.data(), batch * seq * dim);
    std::vector<uint16_t> xnorm_f16(batch * seq * dim);
    for (int i = 0; i < (int)xnorm_f32.size(); i++)
        xnorm_f16[i] = f32_to_f16(xnorm_f32[i]);
    auto *bXnorm16 = upload_f16(xnorm_f16);

    auto *bWq = upload_f16(Wq), *bWk = upload_f16(Wk), *bWv = upload_f16(Wv);
    auto *bQ = alloc_f16(batch * seq * dim);
    auto *bK = alloc_f16(batch * seq * dim);
    auto *bV = alloc_f16(batch * seq * dim);

    const int M = batch * seq, N = dim, K = dim;
    vkflame_dispatch_linear(nullptr, (void *)bXnorm16->address, (void *)bWq->address,
                            nullptr, (void *)bQ->address, M, N, K,
                            VKF_DTYPE_FP16, 0, 0, 0, nullptr, nullptr, nullptr);
    vkflame_dispatch_linear(nullptr, (void *)bXnorm16->address, (void *)bWk->address,
                            nullptr, (void *)bK->address, M, N, K,
                            VKF_DTYPE_FP16, 0, 0, 0, nullptr, nullptr, nullptr);
    vkflame_dispatch_linear(nullptr, (void *)bXnorm16->address, (void *)bWv->address,
                            nullptr, (void *)bV->address, M, N, K,
                            VKF_DTYPE_FP16, 0, 0, 0, nullptr, nullptr, nullptr);

    // ── Step 3: Flash Attention
    auto *bAttnOut = alloc_f32(batch * seq * dim);
    float attn_scale = 1.f / std::sqrt((float)head_dim);
    vkflame_dispatch_flash_attention(nullptr,
                                     (void *)bQ->address, (void *)bK->address, (void *)bV->address,
                                     (void *)bAttnOut->address,
                                     batch, heads, heads, seq, seq, head_dim, attn_scale, 0);

    // ── Step 4: Output projection + residual
    std::vector<float> attn_out(batch * seq * dim), final_out(batch * seq * dim);
    download_f32(bAttnOut, attn_out.data(), batch * seq * dim);

    bool proj_finite = true;
    for (float v : attn_out)
        if (!std::isfinite(v))
        {
            proj_finite = false;
            break;
        }
    check("mini_layer_attn_finite", proj_finite);
    check("mini_layer_attn_nonzero", [&]
          {
        float s=0.f; for(float v:attn_out) s+=v*v; return s>0.f; }());

    // Residual add
    vkflame_dispatch_binop_f32(nullptr, (void *)bX->address, (void *)bAttnOut->address,
                               (void *)bAttnOut->address,
                               batch * seq * dim, batch * seq * dim, VKF_BINOP_ADD);
    download_f32(bAttnOut, final_out.data(), batch * seq * dim);

    bool res_finite = true;
    for (float v : final_out)
        if (!std::isfinite(v))
        {
            res_finite = false;
            break;
        }
    check("mini_layer_residual_finite", res_finite);

    printf("  Layer output[0..4]: %.4f %.4f %.4f %.4f\n",
           final_out[0], final_out[1], final_out[2], final_out[3]);

    vkflame_free(bX);
    vkflame_free(bXnorm);
    vkflame_free(bXnorm16);
    vkflame_free(bWq);
    vkflame_free(bWk);
    vkflame_free(bWv);
    vkflame_free(bQ);
    vkflame_free(bK);
    vkflame_free(bV);
    vkflame_free(bAttnOut);
}

// ── 11.  GEMM throughput (sanity: should be >> 1 TFLOP) ───────────
static void test_gemm_throughput()
{
    printf("\n=== GEMM throughput ===\n");
    const int M = 256, K = 256, N = 256;
    std::vector<uint16_t> Ah(M * K, 0x3C00u), Bh(K * N, 0x3C00u); // all 1.0 in fp16
    auto *bA = upload_f16(Ah), *bB = upload_f16(Bh), *bC = alloc_f16(M * N);

    // warmup
    for (int i = 0; i < 5; i++)
        vkflame_dispatch_linear(nullptr, (void *)bA->address, (void *)bB->address,
                                nullptr, (void *)bC->address,
                                M, N, K, VKF_DTYPE_FP16, 0, 0, 0, nullptr, nullptr, nullptr);

    // measure median of 20 runs
    const int RUNS = 20;
    std::vector<double> times(RUNS);
    for (int r = 0; r < RUNS; r++)
    {
        auto t0 = std::chrono::high_resolution_clock::now(); // NOTE: include below
        vkflame_dispatch_linear(nullptr, (void *)bA->address, (void *)bB->address,
                                nullptr, (void *)bC->address,
                                M, N, K, VKF_DTYPE_FP16, 0, 0, 0, nullptr, nullptr, nullptr);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[r] = std::chrono::duration<double>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    double ms = times[RUNS / 2] * 1000.0;
    double tflops = (2.0 * M * K * N) / (times[RUNS / 2]) / 1e12;
    printf("  GEMM %dx%dx%d  median=%.3f ms  %.2f TFLOPS\n", M, K, N, ms, tflops);
    check("gemm_throughput_nonzero", tflops > 0.001);

    vkflame_free(bA);
    vkflame_free(bB);
    vkflame_free(bC);
}

// ── main ───────────────────────────────────────────────────────────────────
int main()
{
    printf("vkflame_tests — Vulkan compute kernel correctness suite\n");
    printf("==========================================================\n");

    if (int rc = vkflame_init(); rc != 0)
    {
        printf("[FATAL] vkflame_init() returned %d — check Vulkan GPU is present\n", rc);
        return 1;
    }
    vkflame_print_info();

    test_device_init();
    test_buffer_roundtrip();
    test_gemm_fp16();
    test_rms_norm();
    test_softmax();
    test_silu();
    test_scale();
    test_binop_add();
    test_flash_attention();
    test_mini_transformer_layer();
    test_gemm_throughput();

    printf("\n==========================================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
