/*
 * runtime/dispatch.cpp
 * Operator dispatch — bind buffers, fill push constants, submit to GPU.
 *
 * Each dispatch function:
 *  1. Gets the pipeline via vkflame_get_pipeline()
 *  2. Allocates a descriptor set and binds buffers
 *  3. Fills push constants — struct MUST match the GLSL layout(push_constant) block
 *  4. Dispatches with correct workgroup counts (never zero)
 *  5. Submits and waits
 *
 * Push constant structs are defined here alongside their dispatch functions.
 * static_assert checks sizes at compile time so a mismatch fails the build,
 * not silently at runtime with VK_ERROR_DEVICE_LOST.
 */

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  define VK_USE_PLATFORM_WIN32_KHR
#  include <windows.h>
#endif

#include "device.h"
#include "buffer.h"

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

/* ------------------------------------------------------------------ */
/* Internal helpers                                                     */
/* ------------------------------------------------------------------ */

#define VKF_CHECK(call)                                                 \
    do {                                                                \
        VkResult _r = (call);                                           \
        if (_r != VK_SUCCESS) {                                         \
            fprintf(stderr, "[vkflame] VkResult=%d at %s:%d\n",        \
                    (int)_r, __FILE__, __LINE__);                       \
        }                                                               \
    } while (0)

static inline uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

/* Allocate a descriptor set, bind up to 3 storage buffers, return the set */
static VkDescriptorSet bind_buffers(VkDevice device,
                                    VkDescriptorPool pool,
                                    VkDescriptorSetLayout layout,
                                    VkBuffer buf0,           /* binding 0 — always required */
                                    VkDeviceSize size0,
                                    VkBuffer buf1,           /* binding 1 — pass VK_NULL_HANDLE to skip */
                                    VkDeviceSize size1,
                                    VkBuffer buf2,           /* binding 2 — pass VK_NULL_HANDLE to skip */
                                    VkDeviceSize size2)
{
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool     = pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts        = &layout;

    VkDescriptorSet ds;
    VKF_CHECK(vkAllocateDescriptorSets(device, &alloc_info, &ds));

    VkDescriptorBufferInfo buf_infos[3]{};
    VkWriteDescriptorSet   writes[3]{};
    uint32_t write_count = 0;

    auto push_buf = [&](uint32_t binding, VkBuffer buf, VkDeviceSize size) {
        if (buf == VK_NULL_HANDLE) return;
        buf_infos[write_count].buffer = buf;
        buf_infos[write_count].offset = 0;
        buf_infos[write_count].range  = size ? size : VK_WHOLE_SIZE;

        writes[write_count].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[write_count].dstSet          = ds;
        writes[write_count].dstBinding      = binding;
        writes[write_count].descriptorCount = 1;
        writes[write_count].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[write_count].pBufferInfo     = &buf_infos[write_count];
        write_count++;
    };

    push_buf(0, buf0, size0);
    push_buf(1, buf1, size1);
    push_buf(2, buf2, size2);

    vkUpdateDescriptorSets(device, write_count, writes, 0, nullptr);
    return ds;
}

/* Submit a single compute dispatch and wait for completion */
static void submit_and_wait(VkDevice device,
                             VkQueue queue,
                             VkCommandPool pool,
                             VkPipelineLayout layout,
                             VkPipeline pipeline,
                             VkDescriptorSet ds,
                             const void* push_constants,
                             uint32_t pc_size,
                             uint32_t gx, uint32_t gy, uint32_t gz)
{
    /* Allocate command buffer */
    VkCommandBufferAllocateInfo cb_alloc{};
    cb_alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_alloc.commandPool        = pool;
    cb_alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_alloc.commandBufferCount = 1;

    VkCommandBuffer cb;
    VKF_CHECK(vkAllocateCommandBuffers(device, &cb_alloc, &cb));

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKF_CHECK(vkBeginCommandBuffer(cb, &begin));

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout, 0, 1, &ds, 0, nullptr);
    if (push_constants && pc_size > 0)
        vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, pc_size, push_constants);
    vkCmdDispatch(cb, gx, gy, gz);

    VKF_CHECK(vkEndCommandBuffer(cb));

    VkSubmitInfo submit{};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cb;

    VkFence fence;
    VkFenceCreateInfo fence_ci{};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VKF_CHECK(vkCreateFence(device, &fence_ci, nullptr, &fence));

    VKF_CHECK(vkQueueSubmit(queue, 1, &submit, fence));
    VKF_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, pool, 1, &cb);
}

/* Per-frame descriptor pool — simple, one-shot */
static VkDescriptorPool make_pool(VkDevice device, uint32_t max_sets)
{
    VkDescriptorPoolSize pool_size{};
    pool_size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = max_sets * 3;   /* up to 3 bindings per set */

    VkDescriptorPoolCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets       = max_sets;
    ci.poolSizeCount = 1;
    ci.pPoolSizes    = &pool_size;

    VkDescriptorPool pool;
    VKF_CHECK(vkCreateDescriptorPool(device, &ci, nullptr, &pool));
    return pool;
}

/* ------------------------------------------------------------------ */
/* Push constant structs — verified sizes in pipeline.cpp comment      */
/* ------------------------------------------------------------------ */

struct LinearPC {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    float    act_scale;
    uint32_t activation;
};
static_assert(sizeof(LinearPC) == 20, "LinearPC size must be 20");

struct LinearFP16PC {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t activation;
};
static_assert(sizeof(LinearFP16PC) == 16, "LinearFP16PC size must be 16");

struct FlashAttnPC {
    uint32_t B;
    uint32_t Hq;
    uint32_t Hkv;
    uint32_t Sq;
    uint32_t Skv;
    uint32_t D;
    float    scale;
    uint32_t is_causal;
};
static_assert(sizeof(FlashAttnPC) == 32, "FlashAttnPC size must be 32");

struct RmsNormPC {
    uint32_t M;
    uint32_t N;
    float    eps;
};
static_assert(sizeof(RmsNormPC) == 12, "RmsNormPC size must be 12");

struct SoftmaxPC {
    uint32_t M;
    uint32_t N;
};
static_assert(sizeof(SoftmaxPC) == 8, "SoftmaxPC size must be 8");

struct TopkPC {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t largest;
};
static_assert(sizeof(TopkPC) == 16, "TopkPC size must be 16");

struct EmbeddingPC {
    uint32_t V;
    uint32_t D;
    uint32_t B;
};
static_assert(sizeof(EmbeddingPC) == 12, "EmbeddingPC size must be 12");

struct KVCachePC {
    uint32_t seq_pos;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t max_seq;
};
static_assert(sizeof(KVCachePC) == 16, "KVCachePC size must be 16");

/* ------------------------------------------------------------------ */
/* Dtype constants                                                      */
/* ------------------------------------------------------------------ */

#define DTYPE_FP16 1
#define DTYPE_INT8 2
#define DTYPE_FP32 3

/* ------------------------------------------------------------------ */
/* vkflame_dispatch_linear                                              */
/* ------------------------------------------------------------------ */

VKF_API void vkflame_dispatch_linear(
    VKFContext* ctx,
    const void* A, const void* B, const void* /*C_bias*/, void* D,
    int M, int N, int K,
    int dtype, int /*transA*/, int /*transB*/, int activation,
    const void* /*alpha*/, const void* /*beta*/, void* /*stream*/)
{
    if (!ctx || M <= 0 || N <= 0 || K <= 0) return;

    /* Select kernel tier */
    VKFKernelID kid;
    auto& f = ctx->features;
    if (f.has_cooperative_matrix && dtype == DTYPE_FP16 &&
        (int64_t)M * N * K > (1 << 30)) {
        kid = VKF_KERNEL_LINEAR_COOP;
    } else if (f.has_integer_dot_product && dtype == DTYPE_INT8) {
        kid = VKF_KERNEL_LINEAR_INT8;
    } else {
        kid = VKF_KERNEL_LINEAR_FP16;
    }

    VKFPipeline* p = vkflame_get_pipeline(kid);
    if (!p) {
        fprintf(stderr, "[vkflame] dispatch_linear: pipeline %d not ready\n", kid);
        return;
    }

    VKFBuffer* buf_a = vkflame_buf_from_ptr(const_cast<void*>(A));
    VKFBuffer* buf_b = vkflame_buf_from_ptr(const_cast<void*>(B));
    VKFBuffer* buf_d = vkflame_buf_from_ptr(D);
    if (!buf_a || !buf_b || !buf_d) {
        fprintf(stderr, "[vkflame] dispatch_linear: invalid buffer pointer\n");
        return;
    }

    VkDevice device = ctx->device;
    VkDescriptorPool pool = make_pool(device, 1);

    VkDescriptorSet ds = bind_buffers(device, pool, p->ds_layout,
        buf_a->buffer, buf_a->size,
        buf_b->buffer, buf_b->size,
        buf_d->buffer, buf_d->size);

    /* Fill push constants for selected kernel */
    if (kid == VKF_KERNEL_LINEAR_INT8) {
        LinearPC pc{};
        pc.M          = (uint32_t)M;
        pc.N          = (uint32_t)N;
        pc.K          = (uint32_t)K;
        pc.act_scale  = 1.0f / 127.0f;
        pc.activation = (uint32_t)activation;
        submit_and_wait(device, ctx->compute_queue, ctx->command_pool,
                        p->layout, p->pipeline, ds,
                        &pc, sizeof(pc),
                        ceil_div(N, 16), ceil_div(M, 16), 1);
    } else {
        LinearFP16PC pc{};
        pc.M          = (uint32_t)M;
        pc.N          = (uint32_t)N;
        pc.K          = (uint32_t)K;
        pc.activation = (uint32_t)activation;
        submit_and_wait(device, ctx->compute_queue, ctx->command_pool,
                        p->layout, p->pipeline, ds,
                        &pc, sizeof(pc),
                        ceil_div(N, 16), ceil_div(M, 16), 1);
    }

    vkFreeDescriptorSets(device, pool, 1, &ds);
    vkDestroyDescriptorPool(device, pool, nullptr);
}

/* ------------------------------------------------------------------ */
/* vkflame_dispatch_flash_attention                                     */
/* ------------------------------------------------------------------ */

VKF_API void vkflame_dispatch_flash_attention(
    VKFContext* ctx,
    const void* Q, const void* K, const void* V, void* O,
    int B, int Hq, int Hkv, int Sq, int Skv, int D,
    float scale, int is_causal)
{
    if (!ctx) return;

    VKFPipeline* p = vkflame_get_pipeline(VKF_KERNEL_FLASH_ATTENTION);
    if (!p) return;

    VKFBuffer* bq = vkflame_buf_from_ptr(const_cast<void*>(Q));
    VKFBuffer* bk = vkflame_buf_from_ptr(const_cast<void*>(K));
    VKFBuffer* bv = vkflame_buf_from_ptr(const_cast<void*>(V));
    VKFBuffer* bo = vkflame_buf_from_ptr(O);
    if (!bq || !bk || !bv || !bo) return;

    /* For flash attention we need 4 buffers — use binding 0=Q, 1=K, 2=V
       and a second descriptor set or extend to 4 bindings.
       For now bind Q+K as one combined buffer via two separate dispatches,
       or use a layout with 4 bindings. The simplest correct approach: */
    VkDevice device = ctx->device;

    /* Pool with 4 bindings for this kernel */
    VkDescriptorPoolSize pool_size{};
    pool_size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 4;
    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_ci.maxSets       = 1;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes    = &pool_size;
    VkDescriptorPool pool;
    VKF_CHECK(vkCreateDescriptorPool(device, &pool_ci, nullptr, &pool));

    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool     = pool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts        = &p->ds_layout;
    VkDescriptorSet ds;
    VKF_CHECK(vkAllocateDescriptorSets(device, &alloc, &ds));

    VkDescriptorBufferInfo buf_infos[4]{};
    VkBuffer bufs[4] = { bq->buffer, bk->buffer, bv->buffer, bo->buffer };
    VkDeviceSize sizes[4] = { bq->size, bk->size, bv->size, bo->size };
    VkWriteDescriptorSet writes[4]{};
    for (int i = 0; i < 4; i++) {
        buf_infos[i].buffer = bufs[i];
        buf_infos[i].offset = 0;
        buf_infos[i].range  = sizes[i];
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = ds;
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);

    FlashAttnPC pc{};
    pc.B        = (uint32_t)B;
    pc.Hq       = (uint32_t)Hq;
    pc.Hkv      = (uint32_t)Hkv;
    pc.Sq       = (uint32_t)Sq;
    pc.Skv      = (uint32_t)Skv;
    pc.D        = (uint32_t)D;
    pc.scale    = scale;
    pc.is_causal = (uint32_t)is_causal;

    /* One workgroup per (batch, q_head, q_row) */
    submit_and_wait(device, ctx->compute_queue, ctx->command_pool,
                    p->layout, p->pipeline, ds,
                    &pc, sizeof(pc),
                    (uint32_t)Sq, (uint32_t)(B * Hq), 1);

    vkFreeDescriptorSets(device, pool, 1, &ds);
    vkDestroyDescriptorPool(device, pool, nullptr);
}

/* ------------------------------------------------------------------ */
/* vkflame_dispatch_rms_norm                                            */
/* ------------------------------------------------------------------ */

VKF_API void vkflame_dispatch_rms_norm(
    VKFContext* ctx,
    const void* X, const void* gamma, void* Y,
    int M, int N, float eps)
{
    if (!ctx || M <= 0 || N <= 0) return;

    VKFPipeline* p = vkflame_get_pipeline(VKF_KERNEL_RMS_NORM);
    if (!p) return;

    VKFBuffer* bx = vkflame_buf_from_ptr(const_cast<void*>(X));
    VKFBuffer* bg = vkflame_buf_from_ptr(const_cast<void*>(gamma));
    VKFBuffer* by = vkflame_buf_from_ptr(Y);
    if (!bx || !bg || !by) return;

    VkDevice device = ctx->device;
    VkDescriptorPool pool = make_pool(device, 1);
    VkDescriptorSet ds = bind_buffers(device, pool, p->ds_layout,
        bx->buffer, bx->size,
        bg->buffer, bg->size,
        by->buffer, by->size);

    RmsNormPC pc{};
    pc.M   = (uint32_t)M;
    pc.N   = (uint32_t)N;
    pc.eps = eps;

    /* One workgroup per row */
    submit_and_wait(device, ctx->compute_queue, ctx->command_pool,
                    p->layout, p->pipeline, ds,
                    &pc, sizeof(pc),
                    (uint32_t)M, 1, 1);

    vkFreeDescriptorSets(device, pool, 1, &ds);
    vkDestroyDescriptorPool(device, pool, nullptr);
}

/* ------------------------------------------------------------------ */
/* vkflame_dispatch_softmax                                             */
/* ------------------------------------------------------------------ */

VKF_API void vkflame_dispatch_softmax(
    VKFContext* ctx,
    const void* X, void* Y,
    int M, int N)
{
    if (!ctx || M <= 0 || N <= 0) return;

    VKFPipeline* p = vkflame_get_pipeline(VKF_KERNEL_SOFTMAX_ONLINE);
    if (!p) return;

    VKFBuffer* bx = vkflame_buf_from_ptr(const_cast<void*>(X));
    VKFBuffer* by = vkflame_buf_from_ptr(Y);
    if (!bx || !by) return;

    VkDevice device = ctx->device;
    VkDescriptorPool pool = make_pool(device, 1);
    VkDescriptorSet ds = bind_buffers(device, pool, p->ds_layout,
        bx->buffer, bx->size,
        VK_NULL_HANDLE, 0,
        by->buffer, by->size);

    SoftmaxPC pc{};
    pc.M = (uint32_t)M;
    pc.N = (uint32_t)N;

    submit_and_wait(device, ctx->compute_queue, ctx->command_pool,
                    p->layout, p->pipeline, ds,
                    &pc, sizeof(pc),
                    (uint32_t)M, 1, 1);

    vkFreeDescriptorSets(device, pool, 1, &ds);
    vkDestroyDescriptorPool(device, pool, nullptr);
}

/* ------------------------------------------------------------------ */
/* vkflame_dispatch_topk                                                */
/* ------------------------------------------------------------------ */

VKF_API void vkflame_dispatch_topk(
    VKFContext* ctx,
    const void* X, void* values, void* indices,
    int M, int N, int K, int largest)
{
    if (!ctx || M <= 0 || N <= 0 || K <= 0) return;

    VKFPipeline* p = vkflame_get_pipeline(VKF_KERNEL_TOPK);
    if (!p) return;

    VKFBuffer* bx  = vkflame_buf_from_ptr(const_cast<void*>(X));
    VKFBuffer* bv  = vkflame_buf_from_ptr(values);
    VKFBuffer* bi  = vkflame_buf_from_ptr(indices);
    if (!bx || !bv || !bi) return;

    /* topk needs 4 bindings: X, values_out, indices_out — bind values+indices
       at bindings 1 and 2, then X at 0 */
    VkDevice device = ctx->device;
    VkDescriptorPool pool = make_pool(device, 1);
    VkDescriptorSet ds = bind_buffers(device, pool, p->ds_layout,
        bx->buffer, bx->size,
        bv->buffer, bv->size,
        bi->buffer, bi->size);

    TopkPC pc{};
    pc.M       = (uint32_t)M;
    pc.N       = (uint32_t)N;
    pc.K       = (uint32_t)K;
    pc.largest = (uint32_t)largest;

    submit_and_wait(device, ctx->compute_queue, ctx->command_pool,
                    p->layout, p->pipeline, ds,
                    &pc, sizeof(pc),
                    (uint32_t)M, 1, 1);

    vkFreeDescriptorSets(device, pool, 1, &ds);
    vkDestroyDescriptorPool(device, pool, nullptr);
}

/* ------------------------------------------------------------------ */
/* vkflame_dispatch_embedding                                           */
/* ------------------------------------------------------------------ */

VKF_API void vkflame_dispatch_embedding(
    VKFContext* ctx,
    const void* weight, const void* indices, void* out,
    int V, int D, int B)
{
    if (!ctx || V <= 0 || D <= 0 || B <= 0) return;

    VKFPipeline* p = vkflame_get_pipeline(VKF_KERNEL_EMBEDDING);
    if (!p) return;

    VKFBuffer* bw = vkflame_buf_from_ptr(const_cast<void*>(weight));
    VKFBuffer* bi = vkflame_buf_from_ptr(const_cast<void*>(indices));
    VKFBuffer* bo = vkflame_buf_from_ptr(out);
    if (!bw || !bi || !bo) return;

    VkDevice device = ctx->device;
    VkDescriptorPool pool = make_pool(device, 1);
    VkDescriptorSet ds = bind_buffers(device, pool, p->ds_layout,
        bw->buffer, bw->size,
        bi->buffer, bi->size,
        bo->buffer, bo->size);

    EmbeddingPC pc{};
    pc.V = (uint32_t)V;
    pc.D = (uint32_t)D;
    pc.B = (uint32_t)B;

    submit_and_wait(device, ctx->compute_queue, ctx->command_pool,
                    p->layout, p->pipeline, ds,
                    &pc, sizeof(pc),
                    ceil_div((uint32_t)(B * D), 64), 1, 1);

    vkFreeDescriptorSets(device, pool, 1, &ds);
    vkDestroyDescriptorPool(device, pool, nullptr);
}
