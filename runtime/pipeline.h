#pragma once
#include <vulkan/vulkan.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    enum VKFKernelID
    {
        VKF_KERNEL_LINEAR_INT8 = 0,
        VKF_KERNEL_LINEAR_FP16 = 1,
        VKF_KERNEL_LINEAR_COOP = 2,
        VKF_KERNEL_FLASH_ATTENTION = 3,
        VKF_KERNEL_WINOGRAD_F23 = 4,
        VKF_KERNEL_WINOGRAD_F45 = 5,
        VKF_KERNEL_CONV_DIRECT = 6,
        VKF_KERNEL_RMS_NORM = 7,
        VKF_KERNEL_SOFTMAX_ONLINE = 8,
        VKF_KERNEL_ELEMENTWISE = 9,
        VKF_KERNEL_REDUCE = 10,
        VKF_KERNEL_TOPK = 11,
        VKF_KERNEL_SORT_RADIX = 12,
        VKF_KERNEL_EMBEDDING = 13,
        VKF_KERNEL_KVCACHE_UPDATE = 14,
        VKF_KERNEL_LINEAR_FP16_FP32OUT = 15, // fp16 A/B → fp32 D (hipblasGemmEx pattern)
        // ── ggml/Ollama dequantization kernels ────────────────────────
        VKF_KERNEL_DEQUANT_Q4_0 = 16,
        VKF_KERNEL_DEQUANT_Q4_1 = 17,
        VKF_KERNEL_DEQUANT_Q8_0 = 18,
        VKF_KERNEL_DEQUANT_Q5_0 = 19,
        VKF_KERNEL_DEQUANT_Q4_K = 20,
        VKF_KERNEL_DEQUANT_Q5_K = 21,
        VKF_KERNEL_DEQUANT_Q6_K = 22,
        // ── ggml element-wise + utility kernels ──────────────────────
        VKF_KERNEL_ELEMENTWISE_F32 = 23, // silu/gelu/relu/tanh etc.
        VKF_KERNEL_ROPE_NEOX = 24,       // rotary position embeddings
        VKF_KERNEL_SCALE_F32 = 25,       // y = scale*x + bias
        VKF_KERNEL_BINOP_F32 = 26,       // add/mul/sub/div
        VKF_KERNEL_RMS_NORM_F32 = 27,    // fp32 rms norm (no gamma)
        VKF_KERNEL_DEQUANT_Q8_1 = 28,    // Q8_1 activations used in mul_mat_q*_q8_1
        VKF_KERNEL_COUNT = 29
    };

    struct VKFPipeline
    {
        VkPipeline pipeline;
        VkPipelineLayout layout;
        VkDescriptorSetLayout ds_layout;
        uint32_t push_constant_size;
    };

    int vkflame_pipelines_init();
    VKFPipeline *vkflame_get_pipeline(VKFKernelID id);
    void vkflame_pipelines_destroy();

#ifdef __cplusplus
} // extern "C"
#endif
