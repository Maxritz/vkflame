#pragma once
#include "device.h"
#include "buffer.h"
#include <stdint.h>

// Dtype integer codes — must match Python/ctypes side
#define VKF_DTYPE_FP32 0
#define VKF_DTYPE_FP16 1
#define VKF_DTYPE_INT8 2
#define VKF_DTYPE_BF16 3
#define VKF_DTYPE_FP16_FP32OUT 4 // fp16 inputs, fp32 output (hipblasGemmEx pattern)

// Dequantization type codes — passed to vkflame_dispatch_dequant
#define VKF_DEQUANT_Q4_0 0
#define VKF_DEQUANT_Q4_1 1
#define VKF_DEQUANT_Q8_0 2
#define VKF_DEQUANT_Q5_0 3
#define VKF_DEQUANT_Q4_K 4
#define VKF_DEQUANT_Q5_K 5
#define VKF_DEQUANT_Q6_K 6

// Element-wise unary op codes — passed to vkflame_dispatch_elementwise_f32
#define VKF_EW_SILU 0
#define VKF_EW_GELU_ERF 1
#define VKF_EW_RELU 2
#define VKF_EW_TANH 3
#define VKF_EW_GELU_QUICK 4
#define VKF_EW_NEG 5
#define VKF_EW_SIGMOID 6
#define VKF_EW_SQR 7
#define VKF_EW_SQRT 8
#define VKF_EW_EXP 9
#define VKF_EW_LOG 10
#define VKF_EW_ABS 11
#define VKF_EW_SGN 12
#define VKF_EW_HARDSIGMOID 13
#define VKF_EW_HARDSWISH 14

// Binary op codes for vkflame_dispatch_binop_f32
#define VKF_BINOP_ADD 0
#define VKF_BINOP_MUL 1
#define VKF_BINOP_SUB 2
#define VKF_BINOP_DIV 3

#ifdef __cplusplus
extern "C"
{
#endif

    void vkflame_dispatch_linear(
        VKFContext *ctx,
        const void *A, const void *B,
        const void *C, void *D,
        int M, int N, int K,
        int dtype,
        int transA, int transB, int activation,
        const void *alpha, const void *beta, void *stream);

    void vkflame_dispatch_flash_attention(
        VKFContext *ctx,
        const void *Q, const void *K, const void *V, void *O,
        int B, int Hq, int Hkv, int Sq, int Skv, int D,
        float scale, int is_causal);

    void vkflame_dispatch_rms_norm(
        VKFContext *ctx,
        const void *X, const void *gamma, void *Y,
        int M, int N, float eps);

    void vkflame_dispatch_softmax(
        VKFContext *ctx,
        const void *X, void *Y,
        int M, int N);

    void vkflame_dispatch_topk(
        VKFContext *ctx,
        const void *X, void *values, void *indices,
        int M, int N, int K, int largest);

    void vkflame_dispatch_embedding(
        VKFContext *ctx,
        const void *weight, const void *indices, void *out,
        int V, int D, int B);

    // ── New ggml/Ollama dispatch functions ────────────────────────────

    // Dequantize a packed quantized tensor to float32
    void vkflame_dispatch_dequant(
        VKFContext *ctx,
        VKFBuffer *src, VKFBuffer *dst,
        uint32_t n_blocks, int quant_type);

    // Unary element-wise op on float32 data (raw device ptrs via resolve_buf)
    void vkflame_dispatch_elementwise_f32(
        VKFContext *ctx,
        const void *src, void *dst,
        uint32_t n, uint32_t op);

    // Rotary position embeddings (neox/llama format), float32
    void vkflame_dispatch_rope_neox(
        VKFContext *ctx,
        const void *src, const void *pos, void *dst,
        uint32_t ne0, uint32_t ne1, uint32_t n_dims, uint32_t n_seqs,
        float theta_scale, float freq_scale,
        uint32_t p0, uint32_t p1);

    // y = scale * x + bias, float32
    void vkflame_dispatch_scale_f32(
        VKFContext *ctx,
        const void *src, void *dst,
        uint32_t n, float scale, float bias);

    // Element-wise binary op, float32, with broadcast on src1
    void vkflame_dispatch_binop_f32(
        VKFContext *ctx,
        const void *src0, const void *src1, void *dst,
        uint32_t n, uint32_t src1_n, uint32_t op);

    // RMS normalization on float32 (no gamma — ggml variant)
    void vkflame_dispatch_rms_norm_f32(
        VKFContext *ctx,
        const void *src, void *dst,
        uint32_t M, uint32_t N, float eps);

#ifdef __cplusplus
} // extern "C"
#endif
