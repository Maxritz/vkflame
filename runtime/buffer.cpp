#include "buffer.h"
#include "device.h"
#include <cstring>
#include <cstdio>
#include <map> // sorted — needed for upper_bound range lookup
#include <unordered_map>
#include <mutex>
#include <memory>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
// Vulkan Win32 extension structs (VkImportMemoryWin32HandleInfoKHR, etc.)
#include <vulkan/vulkan_win32.h>
// Note: hipIpcGetMemHandle is loaded dynamically at runtime via GetProcAddress
// to avoid a circular link dependency (amdhip64 shim → vkflame_rt).
#endif

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

// Sorted map: key = buffer base address, value = VKFBuffer*.
// Using std::map so we can do upper_bound() range lookups for
// PyTorch sub-allocations (base_ptr + offset) that don't land exactly
// on the registered base address.
static std::map<VkDeviceAddress, VKFBuffer *> g_ptr_map;
static std::mutex g_ptr_map_mutex;

static constexpr size_t ALIGN = 256;

static size_t align_up(size_t n) { return (n + ALIGN - 1) & ~(ALIGN - 1); }

// ── Staging arena (256 MB, permanently mapped) ────────────────────────────
// Eliminates one vkAllocateMemory + vkFreeMemory pair per transfer call.
// On WDDM those driver calls cost ~1–5 ms each; a transformer layer with
// ~80 dispatches would accumulate ~400–800 ms of pure driver overhead.
// Since every transfer function calls submit_and_wait() synchronously, the
// arena can safely be reused from offset 0 on every new call.

static constexpr size_t STAGING_ARENA_BYTES = 256ULL * 1024 * 1024; // 256 MB

static struct StagingArena
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void *mapped = nullptr;
    size_t capacity = 0;
    bool is_coherent = true; // set at init; used to gate vkInvalidateMappedMemoryRanges
} g_staging{};

// ── Command-buffer batching ────────────────────────────────────────────────
// Merges the three vkQueueSubmit+vkWaitForFences calls that make up a single
// _staged() op (h2d upload + compute dispatch + d2h download) into ONE submit.
// This reduces WDDM kernel-mode transition overhead from ~3 ms to ~1 ms per op.
//
// When g_batch_active == true:
//   vkflame_memcpy_h2d  — records copy into g_batch_cb; no submit
//   do_dispatch()       — records compute into g_batch_cb (with TRANSFER→COMPUTE barrier); no submit
//   vkflame_memcpy_d2h  — records copy into g_batch_cb (with COMPUTE→TRANSFER barrier);
//                         stores {cpu_dst, staging_offset, size} in g_batch_d2h_list
// vkflame_batch_end() submits once, waits once, then memcpy's staging → cpu.

static bool g_batch_active = false;
static VkCommandBuffer g_batch_cb = VK_NULL_HANDLE;
static VkFence g_batch_fence = VK_NULL_HANDLE;
static size_t g_batch_stg_off = 0; // next free byte in staging for this batch
// Barrier state: track whether a preceding stage needs a pipeline barrier.
static bool g_batch_need_comp_barrier = false; // h2d done → TRANSFER→COMPUTE needed
static bool g_batch_need_xfer_barrier = false; // compute done → COMPUTE→TRANSFER needed

struct BatchD2H
{
    void *cpu_dst;
    size_t stg_off;
    size_t size;
};
static std::vector<BatchD2H> g_batch_d2h_list;
static std::vector<VkDescriptorPool> g_batch_desc_pools; // deferred-destroy after fence

// Non-extern-C helpers; visible to dispatch.cpp (same shared library).
bool vkf_batch_active() { return g_batch_active; }
VkCommandBuffer vkf_batch_cb() { return g_batch_cb; }

// Insert TRANSFER→COMPUTE barrier if an h2d copy preceded the current dispatch.
void vkf_batch_pre_compute_barrier()
{
    if (!g_batch_active || !g_batch_need_comp_barrier)
        return;
    VkMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(g_batch_cb,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &b, 0, nullptr, 0, nullptr);
    g_batch_need_comp_barrier = false;
    g_batch_need_xfer_barrier = true; // compute follows; d2h will need COMPUTE→TRANSFER
}

// Called by do_dispatch() to hand off its temporary descriptor pool so it
// can be destroyed after the batch fence wait rather than immediately.
void vkf_batch_defer_pool(VkDescriptorPool pool)
{
    g_batch_desc_pools.push_back(pool);
}

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_bits, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++)
    {
        if ((type_bits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return UINT32_MAX;
}

// ── Single command buffer submit + wait ──────────────────────────
static void submit_and_wait(VkDevice device, VkCommandPool pool, VkQueue queue,
                            void (*record)(VkCommandBuffer, void *), void *userdata)
{
    VkCommandBufferAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cb;
    vkAllocateCommandBuffers(device, &ai, &cb);

    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    record(cb, userdata);

    vkEndCommandBuffer(cb);

    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;

    VkFenceCreateInfo fi = {};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    vkCreateFence(device, &fi, nullptr, &fence);
    vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, pool, 1, &cb);
}

// Called once from vkflame_init() after the Vulkan device is ready.
void staging_arena_init()
{
    VKFContext *ctx = vkflame_get_context();
    if (!ctx)
        return;

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = STAGING_ARENA_BYTES;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(ctx->device, &bci, nullptr, &g_staging.buffer) != VK_SUCCESS)
    {
        fprintf(stderr, "[vkflame] staging arena: vkCreateBuffer failed — will use slow path\n");
        return;
    }

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(ctx->device, g_staging.buffer, &mr);

    uint32_t mt = find_memory_type(ctx->physical_device, mr.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mt;

    if (vkAllocateMemory(ctx->device, &mai, nullptr, &g_staging.memory) != VK_SUCCESS)
    {
        fprintf(stderr, "[vkflame] staging arena: vkAllocateMemory failed — will use slow path\n");
        vkDestroyBuffer(ctx->device, g_staging.buffer, nullptr);
        g_staging.buffer = VK_NULL_HANDLE;
        return;
    }

    vkBindBufferMemory(ctx->device, g_staging.buffer, g_staging.memory, 0);
    vkMapMemory(ctx->device, g_staging.memory, 0, VK_WHOLE_SIZE, 0, &g_staging.mapped);
    g_staging.capacity = STAGING_ARENA_BYTES;
    // We explicitly requested HOST_COHERENT, so CPU writes are immediately visible
    // to the GPU without vkFlushMappedMemoryRanges.
    g_staging.is_coherent = true;
    fprintf(stderr, "[vkflame] staging arena: 256 MB permanently mapped\n");
}

// Called from vkflame_shutdown() before device destruction.
void staging_arena_destroy()
{
    VKFContext *ctx = vkflame_get_context();
    if (!ctx || g_staging.buffer == VK_NULL_HANDLE)
        return;
    if (g_staging.mapped)
    {
        vkUnmapMemory(ctx->device, g_staging.memory);
        g_staging.mapped = nullptr;
    }
    vkDestroyBuffer(ctx->device, g_staging.buffer, nullptr);
    vkFreeMemory(ctx->device, g_staging.memory, nullptr);
    g_staging.buffer = VK_NULL_HANDLE;
    g_staging.memory = VK_NULL_HANDLE;
    g_staging.capacity = 0;
}

extern "C"
{

    VKFBuffer *vkflame_alloc(size_t size)
    {
        VKFContext *ctx = vkflame_get_context();
        if (!ctx)
            return nullptr;

        size_t aligned = align_up(size ? size : 1);

        VkBufferCreateInfo bci = {};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = aligned;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        auto *buf = new VKFBuffer{};
        buf->size = aligned;
        buf->is_wrapped = false;

        VKF_CHECK(vkCreateBuffer(ctx->device, &bci, nullptr, &buf->buffer));

        VkMemoryRequirements mem_req;
        vkGetBufferMemoryRequirements(ctx->device, buf->buffer, &mem_req);

        uint32_t mem_type = find_memory_type(ctx->physical_device,
                                             mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkMemoryAllocateFlagsInfo flags_info = {};
        flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo mai = {};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = &flags_info;
        mai.allocationSize = mem_req.size;
        mai.memoryTypeIndex = mem_type;

        VKF_CHECK(vkAllocateMemory(ctx->device, &mai, nullptr, &buf->memory));
        VKF_CHECK(vkBindBufferMemory(ctx->device, buf->buffer, buf->memory, 0));

        VkBufferDeviceAddressInfo addr_info = {};
        addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addr_info.buffer = buf->buffer;
        buf->address = vkGetBufferDeviceAddress(ctx->device, &addr_info);

        {
            std::lock_guard<std::mutex> lock(g_ptr_map_mutex);
            g_ptr_map[buf->address] = buf;
        }
        return buf;
    }

    void vkflame_free(VKFBuffer *buf)
    {
        if (!buf)
            return;
        VKFContext *ctx = vkflame_get_context();
        if (ctx)
        {
            {
                std::lock_guard<std::mutex> lock(g_ptr_map_mutex);
                g_ptr_map.erase(buf->address);
            }
            vkDestroyBuffer(ctx->device, buf->buffer, nullptr);
            // For wrapped (imported) buffers vkFreeMemory only releases the
            // Vulkan reference — the underlying HIP allocation is NOT freed.
            // For owned buffers it frees device memory normally.
            vkFreeMemory(ctx->device, buf->memory, nullptr);
        }
        delete buf;
    }

    VKFBuffer *vkflame_buf_from_ptr(void *ptr)
    {
        if (!ptr)
            return nullptr;
        VkDeviceAddress addr = (VkDeviceAddress)(uintptr_t)ptr;
        std::lock_guard<std::mutex> lock(g_ptr_map_mutex);

        // Fast exact hit — covers the common case (staging buffers, Linux direct).
        auto exact = g_ptr_map.find(addr);
        if (exact != g_ptr_map.end())
        {
            exact->second->current_offset = 0;
            return exact->second;
        }

        // Range lookup: find the largest registered base address <= addr.
        // Handles PyTorch caching-allocator sub-allocations where the tensor
        // data_ptr() is base_ptr + some offset, but only base_ptr is registered.
        auto it = g_ptr_map.upper_bound(addr); // first entry > addr
        if (it == g_ptr_map.begin())
            return nullptr; // addr is below every registered buffer
        --it;

        VKFBuffer *buf = it->second;
        VkDeviceAddress base = it->first;
        if (addr < base || addr >= base + (VkDeviceAddress)buf->size)
            return nullptr; // addr is above this buffer's end

        buf->current_offset = (size_t)(addr - base);
        return buf;
    }

    uint64_t vkflame_buf_address(VKFBuffer *buf)
    {
        return buf ? (uint64_t)buf->address : 0;
    }

    // ── Staging buffer helpers ───────────────────────────────────────

    int vkflame_memcpy_h2d(VKFBuffer *dst, const void *src, size_t size, size_t offset)
    {
        if (!dst || !src)
            return -1;
        VKFContext *ctx = vkflame_get_context();
        if (!ctx)
            return -1;

        // ── Fast path: arena (no per-call vkAllocateMemory) ──────────────
        // HOST_COHERENT memory — no explicit flush required after memcpy.
        if (g_staging.buffer != VK_NULL_HANDLE && size <= g_staging.capacity)
        {
            // ── Batch mode: record into shared CB, no submit yet ──────────
            if (g_batch_active && g_batch_stg_off + size <= g_staging.capacity)
            {
                size_t my_off = g_batch_stg_off;
                memcpy((char *)g_staging.mapped + my_off, src, size);
                g_batch_stg_off = align_up(my_off + size);
                VkBufferCopy region = {(VkDeviceSize)my_off, (VkDeviceSize)offset, (VkDeviceSize)size};
                vkCmdCopyBuffer(g_batch_cb, g_staging.buffer, dst->buffer, 1, &region);
                g_batch_need_comp_barrier = true;
                return 0;
            }
            // ── Normal single-op path ─────────────────────────────────────
            memcpy(g_staging.mapped, src, size);

            struct CB
            {
                VkBuffer src_buf, dst_buf;
                VkDeviceSize size, offset;
            };
            CB cb_data = {g_staging.buffer, dst->buffer,
                          (VkDeviceSize)size, (VkDeviceSize)offset};
            submit_and_wait(ctx->device, ctx->command_pool, ctx->compute_queue, [](VkCommandBuffer cb, void *ud)
                            {
                    auto *c = (CB *)ud;
                    VkBufferCopy region = { 0, c->offset, c->size };
                    vkCmdCopyBuffer(cb, c->src_buf, c->dst_buf, 1, &region); }, &cb_data);
            return 0;
        }

        // ── Slow path: per-call allocation (size > 256 MB) ───────────────
        VkBufferCreateInfo bci = {};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer staging;
        VKF_CHECK(vkCreateBuffer(ctx->device, &bci, nullptr, &staging));

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(ctx->device, staging, &mr);

        uint32_t mt = find_memory_type(ctx->physical_device, mr.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkMemoryAllocateInfo mai = {};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = mt;

        VkDeviceMemory staging_mem;
        VKF_CHECK(vkAllocateMemory(ctx->device, &mai, nullptr, &staging_mem));
        VKF_CHECK(vkBindBufferMemory(ctx->device, staging, staging_mem, 0));

        void *mapped;
        VKF_CHECK(vkMapMemory(ctx->device, staging_mem, 0, size, 0, &mapped));
        memcpy(mapped, src, size);
        vkUnmapMemory(ctx->device, staging_mem);

        struct CB
        {
            VkBuffer src_buf, dst_buf;
            VkDeviceSize size, offset;
        };
        CB cb_data = {staging, dst->buffer, (VkDeviceSize)size, (VkDeviceSize)offset};
        submit_and_wait(ctx->device, ctx->command_pool, ctx->compute_queue, [](VkCommandBuffer cb, void *ud)
                        {
                auto *c = (CB *)ud;
                VkBufferCopy region = { 0, c->offset, c->size };
                vkCmdCopyBuffer(cb, c->src_buf, c->dst_buf, 1, &region); }, &cb_data);

        vkDestroyBuffer(ctx->device, staging, nullptr);
        vkFreeMemory(ctx->device, staging_mem, nullptr);
        return 0;
    }

    int vkflame_memcpy_d2h(void *dst, VKFBuffer *src, size_t size, size_t offset)
    {
        if (!dst || !src)
            return -1;
        VKFContext *ctx = vkflame_get_context();
        if (!ctx)
            return -1;

        // ── Fast path: arena ─────────────────────────────────────────────
        if (g_staging.buffer != VK_NULL_HANDLE && size <= g_staging.capacity)
        {
            // ── Batch mode: record copy, defer memcpy until after fence ────
            if (g_batch_active && g_batch_stg_off + size <= g_staging.capacity)
            {
                // Insert COMPUTE→TRANSFER barrier if a dispatch preceded this d2h.
                if (g_batch_need_xfer_barrier)
                {
                    VkMemoryBarrier b2{};
                    b2.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                    b2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    b2.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    vkCmdPipelineBarrier(g_batch_cb,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         0, 1, &b2, 0, nullptr, 0, nullptr);
                    g_batch_need_xfer_barrier = false;
                }
                size_t my_off = g_batch_stg_off;
                VkBufferCopy region = {(VkDeviceSize)offset, (VkDeviceSize)my_off, (VkDeviceSize)size};
                vkCmdCopyBuffer(g_batch_cb, src->buffer, g_staging.buffer, 1, &region);
                g_batch_stg_off = align_up(my_off + size);
                g_batch_d2h_list.push_back({dst, my_off, size});
                return 0;
            }
            // ── Normal single-op path ─────────────────────────────────────
            struct CB
            {
                VkBuffer src_buf, dst_buf;
                VkDeviceSize size, offset;
            };
            CB cb_data = {src->buffer, g_staging.buffer,
                          (VkDeviceSize)size, (VkDeviceSize)offset};
            submit_and_wait(ctx->device, ctx->command_pool, ctx->compute_queue, [](VkCommandBuffer cb, void *ud)
                            {
                    auto *c = (CB *)ud;
                    VkBufferCopy region = { c->offset, 0, c->size };
                    vkCmdCopyBuffer(cb, c->src_buf, c->dst_buf, 1, &region); }, &cb_data);
            // HOST_COHERENT — data is visible to CPU after the fence wait above.
            memcpy(dst, g_staging.mapped, size);
            return 0;
        }

        // ── Slow path: per-call allocation ───────────────────────────────
        VkBufferCreateInfo bci = {};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer staging;
        VKF_CHECK(vkCreateBuffer(ctx->device, &bci, nullptr, &staging));

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(ctx->device, staging, &mr);

        uint32_t mt = find_memory_type(ctx->physical_device, mr.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkMemoryAllocateInfo mai = {};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = mt;

        VkDeviceMemory staging_mem;
        VKF_CHECK(vkAllocateMemory(ctx->device, &mai, nullptr, &staging_mem));
        VKF_CHECK(vkBindBufferMemory(ctx->device, staging, staging_mem, 0));

        struct CB
        {
            VkBuffer src_buf, dst_buf;
            VkDeviceSize size, offset;
        };
        CB cb_data = {src->buffer, staging, (VkDeviceSize)size, (VkDeviceSize)offset};
        submit_and_wait(ctx->device, ctx->command_pool, ctx->compute_queue, [](VkCommandBuffer cb, void *ud)
                        {
                auto *c = (CB *)ud;
                VkBufferCopy region = { c->offset, 0, c->size };
                vkCmdCopyBuffer(cb, c->src_buf, c->dst_buf, 1, &region); }, &cb_data);

        void *mapped;
        VKF_CHECK(vkMapMemory(ctx->device, staging_mem, 0, size, 0, &mapped));
        memcpy(dst, mapped, size);
        vkUnmapMemory(ctx->device, staging_mem);

        vkDestroyBuffer(ctx->device, staging, nullptr);
        vkFreeMemory(ctx->device, staging_mem, nullptr);
        return 0;
    }

    int vkflame_memcpy_d2d(VKFBuffer *dst, VKFBuffer *src, size_t size)
    {
        if (!dst || !src)
            return -1;
        VKFContext *ctx = vkflame_get_context();
        if (!ctx)
            return -1;

        struct CB
        {
            VkBuffer src_buf, dst_buf;
            VkDeviceSize size;
        };
        CB cb_data = {src->buffer, dst->buffer, (VkDeviceSize)size};

        submit_and_wait(ctx->device, ctx->command_pool, ctx->compute_queue, [](VkCommandBuffer cb, void *ud)
                        {
            auto* c = (CB*)ud;
            VkBufferCopy region = { 0, 0, c->size };
            vkCmdCopyBuffer(cb, c->src_buf, c->dst_buf, 1, &region); }, &cb_data);
        return 0;
    }

    int vkflame_memset(VKFBuffer *buf, int value, size_t size)
    {
        if (!buf)
            return -1;
        VKFContext *ctx = vkflame_get_context();
        if (!ctx)
            return -1;

        struct CB
        {
            VkBuffer buf;
            VkDeviceSize size;
            uint32_t val;
        };
        uint8_t b = (uint8_t)value;
        uint32_t fill = (uint32_t)(b | (b << 8) | (b << 16) | (b << 24));
        CB cb_data = {buf->buffer, (VkDeviceSize)size, fill};

        submit_and_wait(ctx->device, ctx->command_pool, ctx->compute_queue, [](VkCommandBuffer cb, void *ud)
                        {
            auto* c = (CB*)ud;
            vkCmdFillBuffer(cb, c->buf, 0, c->size, c->val); }, &cb_data);
        return 0;
    }

#ifdef _WIN32
    // ── Win32 per-VA-space HIP→Vulkan wrapper map ─────────────────────────
    // Maps HIP device pointer (uint64_t) → VKFBuffer* for allocations made
    // via vkflame_pytorch_malloc so that _staged() can skip staging.
    static std::unordered_map<uint64_t, VKFBuffer *> g_hip_ptr_map;
    static std::mutex g_hip_ptr_mutex;

    // ── Lazy-load the REAL ROCm hip runtime (not our shim) ────────────────
    // We use the versioned "amdhip64_7.dll" which is the actual ROCm library.
    // Avoid "amdhip64.dll" — that's our shim, causing circular dependencies.
    static HMODULE s_real_hip_dll = nullptr;
    static std::once_flag s_hip_dll_flag;

    static HMODULE get_real_hip_dll()
    {
        std::call_once(s_hip_dll_flag, []()
                       {
            // Try versioned name first — this is the real ROCm runtime DLL.
            s_real_hip_dll = LoadLibraryA("amdhip64_7.dll");
            if (!s_real_hip_dll)
            {
                // Last resort: search by full ROCm install path.
                s_real_hip_dll = LoadLibraryA(
                    "C:\\Program Files\\AMD\\ROCm\\7.1\\bin\\amdhip64_7.dll");
            } });
        return s_real_hip_dll;
    }

    // ── Win32 zero-copy: wrap a HIP device pointer in a VkBuffer ──────────
    //
    // Correct API for same-process Vulkan interop: hipMemGetExportHandle
    // (NOT hipIpcGetMemHandle which is for inter-process sharing and fails
    // on caching-allocator sub-allocations with VK_ERROR_DEVICE_LOST).
    //
    // Returns NULL if the extension is absent or the export fails.
    // Callers must fall back to the staging path on NULL.
    //
    // Freeing a wrapped buffer (is_wrapped == true):
    //   vkDestroyBuffer + vkFreeMemory release only the Vulkan-side
    //   reference.  The underlying HIP allocation is *not* freed.

    VKFBuffer *vkflame_wrap_hip_ptr(void *hip_ptr, size_t size)
    {
        VKFContext *ctx = vkflame_get_context();
        if (!ctx || !ctx->features.has_external_memory_win32)
            return nullptr;

        // ── Step 1: export HIP allocation as Win32 KMT handle ──────────
        // hipMemGetExportHandle is the correct same-process export API.
        // hipMemHandleTypeWin32KMT = 1 (from ROCm headers).
        // Signature: hipError_t hipMemGetExportHandle(void* handle,
        //                void* dev_ptr, hipMemRangeHandleType, unsigned long long)
        using PFN_hipMemGetExportHandle = int (*)(void *, const void *, int, unsigned long long);
        static PFN_hipMemGetExportHandle pfn_hipMemGetExportHandle = nullptr;
        if (!pfn_hipMemGetExportHandle)
        {
            HMODULE hip_dll = get_real_hip_dll();
            if (hip_dll)
                pfn_hipMemGetExportHandle =
                    (PFN_hipMemGetExportHandle)GetProcAddress(
                        hip_dll, "hipMemGetExportHandle");
        }
        if (!pfn_hipMemGetExportHandle)
        {
            fprintf(stderr, "[vkflame] hipMemGetExportHandle not found in real HIP dll\n");
            return nullptr;
        }

        HANDLE kmt_handle = nullptr;
        // hipMemHandleTypeWin32KMT = 1
        int hip_err = pfn_hipMemGetExportHandle(&kmt_handle, hip_ptr, 1, 0);
        if (hip_err != 0 /* hipSuccess */)
        {
            // Allocation is not exportable — caller should stage.
            return nullptr;
        }
        if (!kmt_handle)
            return nullptr;

        // ── Step 2: create VkBuffer backed by external Win32 memory ────
        VkExternalMemoryBufferCreateInfo ext_buf_info = {};
        ext_buf_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        ext_buf_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;

        VkBufferCreateInfo bci = {};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.pNext = &ext_buf_info;
        bci.size = size ? size : 1;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        auto *buf = new VKFBuffer{};
        buf->size = size;
        buf->is_wrapped = true;

        if (vkCreateBuffer(ctx->device, &bci, nullptr, &buf->buffer) != VK_SUCCESS)
        {
            delete buf;
            return nullptr;
        }

        VkMemoryRequirements mem_req;
        vkGetBufferMemoryRequirements(ctx->device, buf->buffer, &mem_req);

        // ── Step 3: import the KMT handle into Vulkan device memory ────
        VkImportMemoryWin32HandleInfoKHR import_info = {};
        import_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        import_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
        import_info.handle = kmt_handle;

        // Need DEVICE_ADDRESS_BIT on the allocation to use vkGetBufferDeviceAddress.
        VkMemoryAllocateFlagsInfo flags_info = {};
        flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        flags_info.pNext = &import_info;

        uint32_t mem_type = find_memory_type(ctx->physical_device,
                                             mem_req.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mem_type == UINT32_MAX)
        {
            vkDestroyBuffer(ctx->device, buf->buffer, nullptr);
            delete buf;
            return nullptr;
        }

        VkMemoryAllocateInfo mai = {};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = &flags_info;
        mai.allocationSize = mem_req.size;
        mai.memoryTypeIndex = mem_type;

        if (vkAllocateMemory(ctx->device, &mai, nullptr, &buf->memory) != VK_SUCCESS)
        {
            vkDestroyBuffer(ctx->device, buf->buffer, nullptr);
            delete buf;
            return nullptr;
        }

        if (vkBindBufferMemory(ctx->device, buf->buffer, buf->memory, 0) != VK_SUCCESS)
        {
            vkFreeMemory(ctx->device, buf->memory, nullptr);
            vkDestroyBuffer(ctx->device, buf->buffer, nullptr);
            delete buf;
            return nullptr;
        }

        // ── Step 4: device address for pass-through to dispatch ─────────
        VkBufferDeviceAddressInfo addr_info = {};
        addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addr_info.buffer = buf->buffer;
        buf->address = vkGetBufferDeviceAddress(ctx->device, &addr_info);

        {
            std::lock_guard<std::mutex> lock(g_ptr_map_mutex);
            g_ptr_map[buf->address] = buf;
        }
        return buf;
    }

    // ── Exportable HIP allocator — allocates memory that hipMemGetExportHandle
    // can handle (top-level WDDM resource, not a caching-allocator slice). ──
    //
    // Called by vkflame_pytorch_malloc and by hipMalloc shim (step 3 of plan).
    // Returns a real HIP device pointer, or nullptr on failure.

    void *vkflame_hip_alloc_exportable(size_t size)
    {
        // hipExtMallocWithFlags(ptr, size, hipDeviceMallocDefault=0)
        // creates a device allocation that is a true top-level WDDM resource.
        using PFN_hipExtMallocWithFlags = int (*)(void **, size_t, unsigned int);
        static PFN_hipExtMallocWithFlags pfn = nullptr;
        if (!pfn)
        {
            HMODULE hip_dll = get_real_hip_dll();
            if (hip_dll)
                pfn = (PFN_hipExtMallocWithFlags)GetProcAddress(
                    hip_dll, "hipExtMallocWithFlags");
        }
        if (!pfn)
        {
            // Fallback: standard hipMalloc from real runtime
            using PFN_hipMalloc = int (*)(void **, size_t);
            static PFN_hipMalloc pfn_malloc = nullptr;
            if (!pfn_malloc)
            {
                HMODULE hip_dll = get_real_hip_dll();
                if (hip_dll)
                    pfn_malloc = (PFN_hipMalloc)GetProcAddress(hip_dll, "hipMalloc");
            }
            if (!pfn_malloc)
                return nullptr;
            void *ptr = nullptr;
            if (pfn_malloc(&ptr, size) != 0)
                return nullptr;
            return ptr;
        }
        void *ptr = nullptr;
        if (pfn(&ptr, size, 0 /* hipDeviceMallocDefault */) != 0)
            return nullptr;
        return ptr;
    }

    void vkflame_hip_free_exportable(void *ptr)
    {
        if (!ptr)
            return;
        using PFN_hipFree = int (*)(void *);
        static PFN_hipFree pfn = nullptr;
        if (!pfn)
        {
            HMODULE hip_dll = get_real_hip_dll();
            if (hip_dll)
                pfn = (PFN_hipFree)GetProcAddress(hip_dll, "hipFree");
        }
        if (pfn)
            pfn(ptr);
    }

    // ── PyTorch custom-allocator hooks ─────────────────────────────────────
    // Signature exactly matches torch.cuda.memory.CUDAPluggableAllocator.
    // PyTorch will call  vkflame_pytorch_malloc(nbytes, device, stream)
    // and store the returned void* as tensor.data_ptr().
    //
    // Every allocation is an exportable HIP address that also has a zero-copy
    // VkBuffer registered in g_hip_ptr_map.  When _staged() sees a CUDA tensor
    // whose data_ptr() is in g_hip_ptr_map, it skips staging entirely.

    void *vkflame_pytorch_malloc(size_t size, int /*device*/, void * /*stream*/)
    {
        if (size == 0)
            return nullptr;

        void *hip_ptr = vkflame_hip_alloc_exportable(size);
        if (!hip_ptr)
            return nullptr;

        // Try to create a zero-copy VkBuffer over this allocation.
        // If it fails we still return the valid HIP pointer — PyTorch can use
        // the tensor, but _staged() will fall back to the staging path.
        VKFBuffer *buf = vkflame_wrap_hip_ptr(hip_ptr, size);
        if (buf)
        {
            std::lock_guard<std::mutex> lk(g_hip_ptr_mutex);
            g_hip_ptr_map[(uint64_t)(uintptr_t)hip_ptr] = buf;
        }
        return hip_ptr;
    }

    void vkflame_pytorch_free(void *ptr, size_t /*size*/, int /*device*/, void * /*stream*/)
    {
        if (!ptr)
            return;

        VKFBuffer *buf = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_hip_ptr_mutex);
            auto it = g_hip_ptr_map.find((uint64_t)(uintptr_t)ptr);
            if (it != g_hip_ptr_map.end())
            {
                buf = it->second;
                g_hip_ptr_map.erase(it);
            }
        }
        // Release Vulkan view (does NOT free the HIP allocation).
        if (buf)
        {
            VKFContext *ctx = vkflame_get_context();
            if (ctx)
            {
                vkDestroyBuffer(ctx->device, buf->buffer, nullptr);
                vkFreeMemory(ctx->device, buf->memory, nullptr);
            }
            delete buf;
        }
        // Free the actual HIP allocation via the real runtime.
        vkflame_hip_free_exportable(ptr);
    }

    // VkBuffer device-address lookup for a HIP data_ptr().
    // Returns 0 if not found (caller should stage).
    uint64_t vkflame_buf_from_hip_ptr(uint64_t hip_ptr)
    {
        std::lock_guard<std::mutex> lk(g_hip_ptr_mutex);
        auto it = g_hip_ptr_map.find(hip_ptr);
        if (it == g_hip_ptr_map.end())
            return 0;
        return (uint64_t)it->second->address;
    }

#endif // _WIN32

    // ── Command-buffer batching ──────────────────────────────────────────

    void vkflame_batch_begin()
    {
        VKFContext *ctx = vkflame_get_context();
        if (!ctx || g_batch_active)
            return;

        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = ctx->command_pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        vkAllocateCommandBuffers(ctx->device, &ai, &g_batch_cb);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(g_batch_cb, &bi);

        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(ctx->device, &fi, nullptr, &g_batch_fence);

        g_batch_active = true;
        g_batch_stg_off = 0;
        g_batch_need_comp_barrier = false;
        g_batch_need_xfer_barrier = false;
        g_batch_d2h_list.clear();
        g_batch_desc_pools.clear();
    }

    void vkflame_batch_end()
    {
        if (!g_batch_active || g_batch_cb == VK_NULL_HANDLE)
            return;
        VKFContext *ctx = vkflame_get_context();
        if (!ctx)
            return;

        vkEndCommandBuffer(g_batch_cb);

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &g_batch_cb;
        VKF_CHECK(vkQueueSubmit(ctx->compute_queue, 1, &si, g_batch_fence));
        VKF_CHECK(vkWaitForFences(ctx->device, 1, &g_batch_fence, VK_TRUE, UINT64_MAX));

        // HOST_COHERENT guard (staging is always coherent here, but defensive).
        if (!g_staging.is_coherent && g_staging.memory != VK_NULL_HANDLE)
        {
            VkMappedMemoryRange inv{};
            inv.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            inv.memory = g_staging.memory;
            inv.offset = 0;
            inv.size = VK_WHOLE_SIZE;
            vkInvalidateMappedMemoryRanges(ctx->device, 1, &inv);
        }

        // Flush deferred d2h copies: staging memory → CPU pointers.
        for (const auto &d : g_batch_d2h_list)
            memcpy(d.cpu_dst, (const char *)g_staging.mapped + d.stg_off, d.size);

        // Destroy descriptor pools that were deferred by do_dispatch() in batch mode.
        for (auto pool : g_batch_desc_pools)
            vkDestroyDescriptorPool(ctx->device, pool, nullptr);

        vkDestroyFence(ctx->device, g_batch_fence, nullptr);
        vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &g_batch_cb);
        g_batch_cb = VK_NULL_HANDLE;
        g_batch_fence = VK_NULL_HANDLE;
        g_batch_active = false;
        g_batch_d2h_list.clear();
        g_batch_desc_pools.clear();
    }

} // extern "C"
