#pragma once
#include <vulkan/vulkan.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    struct VKFBuffer
    {
        VkBuffer buffer;
        VkDeviceMemory memory;
        VkDeviceAddress address; // raw GPU pointer == hipMalloc return value
        size_t size;
        bool is_wrapped; // true = imported from HIP; free() skips vkFreeMemory
    };

    VKFBuffer *vkflame_alloc(size_t size);
    void vkflame_free(VKFBuffer *buf);
    VKFBuffer *vkflame_buf_from_ptr(void *ptr);
    uint64_t vkflame_buf_address(VKFBuffer *buf);

#ifdef _WIN32
    // Zero-copy: wrap an existing HIP device pointer in a VkBuffer via
    // VK_KHR_external_memory_win32 + hipIpcGetMemHandle.
    // Returns NULL if the extension is unavailable or handle export fails
    // (caller should fall back to staged path).
    VKFBuffer *vkflame_wrap_hip_ptr(void *hip_ptr, size_t size);
#endif

    int vkflame_memcpy_h2d(VKFBuffer *dst, const void *src, size_t size, size_t offset);
    int vkflame_memcpy_d2h(void *dst, VKFBuffer *src, size_t size, size_t offset);
    int vkflame_memcpy_d2d(VKFBuffer *dst, VKFBuffer *src, size_t size);
    int vkflame_memset(VKFBuffer *buf, int value, size_t size);

#ifdef __cplusplus
} // extern "C"
#endif
