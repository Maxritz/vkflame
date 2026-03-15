#include "dispatch.h"
#include "device.h"
#include "buffer.h"
#include "pipeline.h"
#include <cstring>
#include <cstdio>
#include <vector>
#include <vulkan/vulkan.h>

#define VKF_CHECK(call)                                         \
    do                                                          \
    {                                                           \
        VkResult _r = (call);                                   \
        if (_r != VK_SUCCESS)                                   \
        {                                                       \
            fprintf(stderr, "[vkflame] VkResult=%d at %s:%d\n", \
                    (int)_r, __FILE__, __LINE__);               \
        }                                                       \
    } while (0)

// ── Dtype constants (must match Python side) ─────────────────────
// DTYPE_FP32 = 0, DTYPE_FP16 = 1, DTYPE_INT8 = 2, DTYPE_BF16 = 3, DTYPE_FP16_FP32OUT = 4
#define DTYPE_FP32 0
#define DTYPE_FP16 1
#define DTYPE_INT8 2
#define DTYPE_BF16 3
#define DTYPE_FP16_FP32OUT 4

// ── Push constant layouts matching each GLSL shader ─────────────

struct LinearInt8PC
{
    uint32_t M, N, K;
    float act_scale;
    uint32_t activation; // 0=none,1=silu,2=relu,3=gelu_erf
};

struct LinearFP16PC
{
    uint32_t M, N, K;
    uint32_t activation;
};

struct FlashAttnPC
{
    uint32_t B;
    uint32_t Hq;
    uint32_t Hkv;
    uint32_t Sq;
    uint32_t Skv;
    uint32_t D;
    float scale;
    uint32_t is_causal;
};

struct RmsNormPC
{
    uint32_t M;
    uint32_t N;
    float eps;
};

struct SoftmaxPC
{
    uint32_t M;
    uint32_t N;
};

struct TopkPC
{
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t largest;
};

struct EmbeddingPC
{
    uint32_t V;
    uint32_t D;
    uint32_t B;
};

// ── Helper: one-shot compute dispatch ────────────────────────────

struct DispatchArgs
{
    VKFKernelID kernel_id;
    const void *push_constants;
    uint32_t pc_size;
    // Buffer bindings in binding order
    std::vector<VkBuffer> buffers;
    std::vector<VkDeviceSize> offsets;
    std::vector<VkDeviceSize> ranges;
    // Dispatch dimensions
    uint32_t gx, gy, gz;
};

static void do_dispatch(const DispatchArgs &args)
{
    VKFContext *ctx = vkflame_get_context();
    VKFPipeline *pipe = vkflame_get_pipeline(args.kernel_id);
    if (!pipe || !pipe->pipeline)
    {
        fprintf(stderr, "[vkflame] pipeline not ready for kernel %d\n", (int)args.kernel_id);
        return;
    }

    VkDevice dev = ctx->device;

    // ── Allocate command buffer ────────────────────────────────
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = ctx->command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    VkCommandBuffer cmd{};
    VKF_CHECK(vkAllocateCommandBuffers(dev, &alloc_info, &cmd));

    // ── Begin recording ────────────────────────────────────────
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKF_CHECK(vkBeginCommandBuffer(cmd, &begin_info));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);

    // ── Descriptor set ─────────────────────────────────────────
    // Allocate from a temporary descriptor pool
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = (uint32_t)args.buffers.size();

    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets = 1;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes = &pool_size;
    pool_ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    VkDescriptorPool desc_pool{};
    VKF_CHECK(vkCreateDescriptorPool(dev, &pool_ci, nullptr, &desc_pool));

    VkDescriptorSetAllocateInfo ds_alloc{};
    ds_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_alloc.descriptorPool = desc_pool;
    ds_alloc.descriptorSetCount = 1;
    ds_alloc.pSetLayouts = &pipe->ds_layout;
    VkDescriptorSet ds{};
    VKF_CHECK(vkAllocateDescriptorSets(dev, &ds_alloc, &ds));

    // Write bindings
    std::vector<VkDescriptorBufferInfo> buf_infos(args.buffers.size());
    std::vector<VkWriteDescriptorSet> writes(args.buffers.size());
    for (uint32_t i = 0; i < (uint32_t)args.buffers.size(); i++)
    {
        buf_infos[i].buffer = args.buffers[i];
        buf_infos[i].offset = args.offsets.empty() ? 0 : args.offsets[i];
        buf_infos[i].range = args.ranges.empty() ? VK_WHOLE_SIZE : args.ranges[i];

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(dev, (uint32_t)writes.size(), writes.data(), 0, nullptr);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe->layout, 0, 1, &ds, 0, nullptr);

    // ── Push constants ─────────────────────────────────────────
    if (args.push_constants && args.pc_size > 0)
    {
        vkCmdPushConstants(cmd, pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, args.pc_size, args.push_constants);
    }

    // ── Dispatch ───────────────────────────────────────────────
    vkCmdDispatch(cmd, args.gx, args.gy, args.gz);

    VKF_CHECK(vkEndCommandBuffer(cmd));

    // ── Submit and wait ────────────────────────────────────────
    VkFenceCreateInfo fence_ci{};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence{};
    VKF_CHECK(vkCreateFence(dev, &fence_ci, nullptr, &fence));

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VKF_CHECK(vkQueueSubmit(ctx->compute_queue, 1, &submit, fence));
    VKF_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

    // ── Cleanup ────────────────────────────────────────────────
    vkDestroyFence(dev, fence, nullptr);
    vkFreeDescriptorSets(dev, desc_pool, 1, &ds);
    vkDestroyDescriptorPool(dev, desc_pool, nullptr);
    vkFreeCommandBuffers(dev, ctx->command_pool, 1, &cmd);
}

// Helper: look up VKFBuffer from a raw device pointer
static VkBuffer resolve_buf(const void *ptr)
{
    if (!ptr)
        return VK_NULL_HANDLE;
    VKFBuffer *b = vkflame_buf_from_ptr(const_cast<void *>(ptr));
    return b ? b->buffer : VK_NULL_HANDLE;
}

// ── Public dispatch API ───────────────────────────────────────────

extern "C"
{

    void vkflame_dispatch_linear(
        VKFContext *ctx,
        const void *A, const void *B_mat,
        const void *C, void *D,
        int M, int N, int K,
        int dtype,
        int transA, int transB, int activation,
        const void *alpha, const void *beta, void *stream)
    {
        (void)ctx;
        (void)transA;
        (void)transB;
        (void)alpha;
        (void)beta;
        (void)stream;

        VKFContext *g_ctx = vkflame_get_context();
        const VKFFeatures &feat = g_ctx->features;

        // Kernel selection
        VKFKernelID kid;
        if (feat.has_cooperative_matrix && dtype == DTYPE_FP16 &&
            (int64_t)M * N * K > (1 << 30))
            kid = VKF_KERNEL_LINEAR_COOP;
        else if (feat.has_integer_dot_product && dtype == DTYPE_INT8)
            kid = VKF_KERNEL_LINEAR_INT8;
        else if (dtype == DTYPE_FP16_FP32OUT)
            kid = VKF_KERNEL_LINEAR_FP16_FP32OUT;
        else
            kid = VKF_KERNEL_LINEAR_FP16;

        // Declare both pc structs here so they outlive do_dispatch().
        // (Storing &pc inside an if/else block and calling do_dispatch after
        //  the block is a dangling-pointer bug → GPU crash / device lost.)
        LinearInt8PC pc_i8{};
        LinearFP16PC pc_f16{};

        DispatchArgs args{};
        args.kernel_id = kid;
        args.buffers = {resolve_buf(A), resolve_buf(B_mat),
                        resolve_buf(C ? C : D), resolve_buf(D)};

        if (kid == VKF_KERNEL_LINEAR_INT8)
        {
            pc_i8.M = (uint32_t)M;
            pc_i8.N = (uint32_t)N;
            pc_i8.K = (uint32_t)K;
            pc_i8.act_scale = 1.0f; // caller may pass via alpha
            pc_i8.activation = (uint32_t)activation;
            args.push_constants = &pc_i8;
            args.pc_size = sizeof(pc_i8);
            // Add scale buffer (binding 4) — re-use D buffer if no separate scale
            args.buffers.push_back(resolve_buf(D)); // placeholder for w_scale
        }
        else
        {
            pc_f16.M = (uint32_t)M;
            pc_f16.N = (uint32_t)N;
            pc_f16.K = (uint32_t)K;
            pc_f16.activation = (uint32_t)activation;
            args.push_constants = &pc_f16;
            args.pc_size = sizeof(pc_f16);
        }

        // Dispatch: one thread per output element; workgroup 16x16
        args.gx = ((uint32_t)N + 15u) / 16u;
        args.gy = ((uint32_t)M + 15u) / 16u;
        args.gz = 1;

        do_dispatch(args);
    }

    void vkflame_dispatch_flash_attention(
        VKFContext *ctx,
        const void *Q, const void *K_ptr, const void *V_ptr, void *O_ptr,
        int B, int Hq, int Hkv, int Sq, int Skv, int D,
        float scale, int is_causal)
    {
        (void)ctx;

        FlashAttnPC pc{};
        pc.B = (uint32_t)B;
        pc.Hq = (uint32_t)Hq;
        pc.Hkv = (uint32_t)Hkv;
        pc.Sq = (uint32_t)Sq;
        pc.Skv = (uint32_t)Skv;
        pc.D = (uint32_t)D;
        pc.scale = scale;
        pc.is_causal = (uint32_t)is_causal;

        DispatchArgs args{};
        args.kernel_id = VKF_KERNEL_FLASH_ATTENTION;
        args.push_constants = &pc;
        args.pc_size = sizeof(pc);
        args.buffers = {resolve_buf(Q), resolve_buf(K_ptr),
                        resolve_buf(V_ptr), resolve_buf(O_ptr)};
        // One workgroup per query position per head per batch
        args.gx = (uint32_t)Sq;
        args.gy = (uint32_t)Hq;
        args.gz = (uint32_t)B;

        do_dispatch(args);
    }

    void vkflame_dispatch_rms_norm(
        VKFContext *ctx,
        const void *X, const void *gamma, void *Y,
        int M, int N, float eps)
    {
        (void)ctx;

        RmsNormPC pc{};
        pc.M = (uint32_t)M;
        pc.N = (uint32_t)N;
        pc.eps = eps;

        DispatchArgs args{};
        args.kernel_id = VKF_KERNEL_RMS_NORM;
        args.push_constants = &pc;
        args.pc_size = sizeof(pc);
        args.buffers = {resolve_buf(X), resolve_buf(gamma), resolve_buf(Y)};
        // One workgroup per row
        args.gx = (uint32_t)M;
        args.gy = 1;
        args.gz = 1;

        do_dispatch(args);
    }

    void vkflame_dispatch_softmax(
        VKFContext *ctx,
        const void *X, void *Y,
        int M, int N)
    {
        (void)ctx;

        SoftmaxPC pc{};
        pc.M = (uint32_t)M;
        pc.N = (uint32_t)N;

        DispatchArgs args{};
        args.kernel_id = VKF_KERNEL_SOFTMAX_ONLINE;
        args.push_constants = &pc;
        args.pc_size = sizeof(pc);
        args.buffers = {resolve_buf(X), resolve_buf(Y)};
        // One workgroup per row
        args.gx = (uint32_t)M;
        args.gy = 1;
        args.gz = 1;

        do_dispatch(args);
    }

    void vkflame_dispatch_topk(
        VKFContext *ctx,
        const void *X, void *values, void *indices,
        int M, int N, int K, int largest)
    {
        (void)ctx;

        TopkPC pc{};
        pc.M = (uint32_t)M;
        pc.N = (uint32_t)N;
        pc.K = (uint32_t)K;
        pc.largest = (uint32_t)largest;

        DispatchArgs args{};
        args.kernel_id = VKF_KERNEL_TOPK;
        args.push_constants = &pc;
        args.pc_size = sizeof(pc);
        args.buffers = {resolve_buf(X), resolve_buf(values), resolve_buf(indices)};
        // One workgroup per row
        args.gx = (uint32_t)M;
        args.gy = 1;
        args.gz = 1;

        do_dispatch(args);
    }

    void vkflame_dispatch_embedding(
        VKFContext *ctx,
        const void *weight, const void *indices, void *out,
        int V, int D, int Batch)
    {
        (void)ctx;

        EmbeddingPC pc{};
        pc.V = (uint32_t)V;
        pc.D = (uint32_t)D;
        pc.B = (uint32_t)Batch;

        DispatchArgs args{};
        args.kernel_id = VKF_KERNEL_EMBEDDING;
        args.push_constants = &pc;
        args.pc_size = sizeof(pc);
        args.buffers = {resolve_buf(weight), resolve_buf(indices), resolve_buf(out)};
        // One thread per output element; workgroup 256
        uint32_t total = (uint32_t)Batch * (uint32_t)D;
        args.gx = (total + 255u) / 256u;
        args.gy = 1;
        args.gz = 1;

        do_dispatch(args);
    }

    // ── New: dequantization dispatcher ───────────────────────────
    // quant_type: 0=Q4_0, 1=Q4_1, 2=Q8_0, 3=Q5_0, 4=Q4_K, 5=Q5_K, 6=Q6_K
    void vkflame_dispatch_dequant(
        VKFContext *ctx,
        VKFBuffer *src, VKFBuffer *dst,
        uint32_t n_blocks, int quant_type)
    {
        (void)ctx;
        static const VKFKernelID kid_map[] = {
            VKF_KERNEL_DEQUANT_Q4_0, VKF_KERNEL_DEQUANT_Q4_1,
            VKF_KERNEL_DEQUANT_Q8_0, VKF_KERNEL_DEQUANT_Q5_0,
            VKF_KERNEL_DEQUANT_Q4_K, VKF_KERNEL_DEQUANT_Q5_K,
            VKF_KERNEL_DEQUANT_Q6_K};
        if (quant_type < 0 || quant_type > 6)
            return;

        struct DequantPC
        {
            uint32_t nb;
        } pc{n_blocks};

        bool small_grid = (quant_type == 0 || quant_type == 1 || quant_type == 3); // Q4_0/Q4_1/Q5_0
        DispatchArgs args{};
        args.kernel_id = kid_map[quant_type];
        args.push_constants = &pc;
        args.pc_size = sizeof(pc);
        args.buffers = {src->buffer, dst->buffer};
        args.gx = small_grid ? (n_blocks + 7u) / 8u : n_blocks;
        args.gy = 1;
        args.gz = 1;
        do_dispatch(args);
    }

    // ── New: elementwise fp32 (silu/gelu/relu/tanh/etc.) ─────────
    // op: 0=silu, 1=gelu_erf, 2=relu, 3=tanh, 4=neg, 5=abs, 6=sqr, 7=sqrt, 8=log
    void vkflame_dispatch_elementwise_f32(
        VKFContext *ctx,
        const void *src, void *dst,
        uint32_t n, uint32_t op)
    {
        (void)ctx;
        struct ElemPC
        {
            uint32_t n;
            uint32_t op;
        } pc{n, op};

        DispatchArgs args{};
        args.kernel_id = VKF_KERNEL_ELEMENTWISE_F32;
        args.push_constants = &pc;
        args.pc_size = sizeof(pc);
        args.buffers = {resolve_buf(src), resolve_buf(dst)};
        args.gx = (n + 255u) / 256u;
        args.gy = 1;
        args.gz = 1;
        do_dispatch(args);
    }

    // ── New: RoPE neox ───────────────────────────────────────────
    void vkflame_dispatch_rope_neox(
        VKFContext *ctx,
        const void *src, const void *pos_buf, void *dst,
        uint32_t ne0, uint32_t ne1, uint32_t n_dims, uint32_t n_seqs,
        float theta_scale, float freq_scale, uint32_t p0, uint32_t p1)
    {
        (void)ctx;
        struct RopePC
        {
            uint32_t ne0, ne1, n_dims, n_seqs;
            float theta_scale, freq_scale;
            uint32_t p0, p1;
        } pc{ne0, ne1, n_dims, n_seqs, theta_scale, freq_scale, p0, p1};

        DispatchArgs args{};
        args.kernel_id = VKF_KERNEL_ROPE_NEOX;
        args.push_constants = &pc;
        args.pc_size = sizeof(pc);
        args.buffers = {resolve_buf(src), resolve_buf(pos_buf), resolve_buf(dst)};
        // One workgroup per (seq, head) pair: ne1 heads × n_seqs sequences
        args.gx = ne1;
        args.gy = n_seqs;
        args.gz = 1;
        do_dispatch(args);
    }

    // ── New: scale_f32  y = scale * x + bias ─────────────────────
    void vkflame_dispatch_scale_f32(
        VKFContext *ctx,
        const void *src, void *dst,
        uint32_t n, float scale, float bias)
    {
        (void)ctx;
        struct ScalePC
        {
            uint32_t n;
            float scale;
            float bias;
        } pc{n, scale, bias};

        DispatchArgs args{};
        args.kernel_id = VKF_KERNEL_SCALE_F32;
        args.push_constants = &pc;
        args.pc_size = sizeof(pc);
        args.buffers = {resolve_buf(src), resolve_buf(dst)};
        args.gx = (n + 255u) / 256u;
        args.gy = 1;
        args.gz = 1;
        do_dispatch(args);
    }

    // ── New: binary op fp32 (add/mul/sub/div with broadcast) ─────
    // op: 0=add, 1=mul, 2=sub, 3=div
    void vkflame_dispatch_binop_f32(
        VKFContext *ctx,
        const void *src0, const void *src1, void *dst,
        uint32_t n, uint32_t src1_n, uint32_t op)
    {
        (void)ctx;
        struct BinopPC
        {
            uint32_t n, op, src1_n, pad;
        } pc{n, op, src1_n, 0};

        DispatchArgs args{};
        args.kernel_id = VKF_KERNEL_BINOP_F32;
        args.push_constants = &pc;
        args.pc_size = sizeof(pc);
        args.buffers = {resolve_buf(src0), resolve_buf(src1), resolve_buf(dst)};
        args.gx = (n + 255u) / 256u;
        args.gy = 1;
        args.gz = 1;
        do_dispatch(args);
    }

    // ── New: rms_norm_f32 (no gamma, fp32 in/out) ────────────────
    void vkflame_dispatch_rms_norm_f32(
        VKFContext *ctx,
        const void *src, void *dst,
        uint32_t M, uint32_t N, float eps)
    {
        (void)ctx;
        struct RmsF32PC
        {
            uint32_t M, N;
            float eps;
        } pc{M, N, eps};

        DispatchArgs args{};
        args.kernel_id = VKF_KERNEL_RMS_NORM_F32;
        args.push_constants = &pc;
        args.pc_size = sizeof(pc);
        args.buffers = {resolve_buf(src), resolve_buf(dst)};
        // One workgroup per row
        args.gx = M;
        args.gy = 1;
        args.gz = 1;
        do_dispatch(args);
    }

} // extern "C"
