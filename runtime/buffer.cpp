#include "buffer.h"
#include "device.h"
#include <cstring>
#include <cstdio>
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

#ifdef _WIN32
    // ── Win32 zero-copy: wrap a HIP device pointer in a VkBuffer ──────────
    //
    // Flow:
    //   1. hipIpcGetMemHandle(hip_ptr) → 64-byte opaque blob.
    //      On AMD ROCm/Windows (WDDM), the first sizeof(HANDLE) bytes
    //      encode the Win32 KMT (Kernel-Mode Thunk) handle for the
    //      underlying D3D/KMD allocation.
    //   2. VkImportMemoryWin32HandleInfoKHR  pNext'd into vkAllocateMemory
    //      imports that KMT handle — same physical pages, zero staging.
    //   3. VkBuffer bound to the imported VkDeviceMemory.
    //   4. vkGetBufferDeviceAddress → usable as dispatch argument.
    //
    // Returns NULL if the extension is absent or the handle export fails
    // (e.g. PyTorch sub-allocates inside a larger pool allocation).
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
        // Load hipIpcGetMemHandle at runtime to avoid a circular link
        // dependency (amdhip64.dll shim already links against vkflame_rt).
        using PFN_hipIpcGetMemHandle = int (*)(void *, void *);
        static PFN_hipIpcGetMemHandle pfn_hipIpcGetMemHandle = nullptr;
        if (!pfn_hipIpcGetMemHandle)
        {
            HMODULE hip_dll = LoadLibraryA("amdhip64.dll");
            if (!hip_dll)
                hip_dll = LoadLibraryA("amdhip64_7.dll");
            if (hip_dll)
                pfn_hipIpcGetMemHandle =
                    (PFN_hipIpcGetMemHandle)GetProcAddress(hip_dll, "hipIpcGetMemHandle");
        }
        if (!pfn_hipIpcGetMemHandle)
            return nullptr;

        // hipIpcMemHandle_t is a 64-byte opaque blob; use a plain char array
        // so we don't need the HIP struct definition for the IPC part.
        char ipc_blob[64] = {};
        int hip_err = pfn_hipIpcGetMemHandle(ipc_blob, hip_ptr);
        if (hip_err != 0 /* hipSuccess */)
        {
            // Not every allocation is IPC-exportable (e.g. PyTorch caching
            // allocator sub-allocations).  Fail silently; caller stages.
            return nullptr;
        }
        // On AMD ROCm/Windows (WDDM), the first sizeof(HANDLE) bytes of the
        // IPC blob contain the Win32 KMT handle for the underlying allocation.
        HANDLE kmt_handle = 0;
        memcpy(&kmt_handle, ipc_blob, sizeof(HANDLE));
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
#endif // _WIN32

} // extern "C"
