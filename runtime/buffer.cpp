#include "buffer.h"
#include "device.h"
#include <cstring>
#include <cstdio>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <algorithm>

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

static std::unordered_map<VkDeviceAddress, VKFBuffer *> g_ptr_map;
static std::mutex g_ptr_map_mutex;

static constexpr size_t ALIGN = 256;

static size_t align_up(size_t n) { return (n + ALIGN - 1) & ~(ALIGN - 1); }

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
            vkFreeMemory(ctx->device, buf->memory, nullptr);
        }
        delete buf;
    }

    VKFBuffer *vkflame_buf_from_ptr(void *ptr)
    {
        VkDeviceAddress addr = (VkDeviceAddress)(uintptr_t)ptr;
        std::lock_guard<std::mutex> lock(g_ptr_map_mutex);
        auto it = g_ptr_map.find(addr);
        return (it != g_ptr_map.end()) ? it->second : nullptr;
    }

    uint64_t vkflame_buf_address(VKFBuffer *buf)
    {
        return buf ? (uint64_t)buf->address : 0;
    }

    // ── Staging buffer helpers ───────────────────────────────────────
    struct CopyH2D
    {
        VkBuffer dst;
        const void *src;
        size_t size;
        size_t offset;
    };

    static void record_h2d(VkCommandBuffer cb, void *ud)
    {
        auto *c = (CopyH2D *)ud;
        VkBufferCopy region = {0, c->offset, c->size};
        vkCmdCopyBuffer(cb, c->dst /* use staging as source via pointer swap */, c->dst, 1, &region);
    }

    int vkflame_memcpy_h2d(VKFBuffer *dst, const void *src, size_t size, size_t offset)
    {
        if (!dst || !src)
            return -1;
        VKFContext *ctx = vkflame_get_context();
        if (!ctx)
            return -1;

        // Create staging buffer
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

        // Copy staging → device
        struct CB
        {
            VkBuffer src_buf, dst_buf;
            VkDeviceSize size, offset;
        };
        CB cb_data = {staging, dst->buffer, (VkDeviceSize)size, (VkDeviceSize)offset};

        submit_and_wait(ctx->device, ctx->command_pool, ctx->compute_queue, [](VkCommandBuffer cb, void *ud)
                        {
            auto* c = (CB*)ud;
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
            auto* c = (CB*)ud;
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

} // extern "C"
