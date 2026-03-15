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
    };

    VKFBuffer *vkflame_alloc(size_t size);
    void vkflame_free(VKFBuffer *buf);
    VKFBuffer *vkflame_buf_from_ptr(void *ptr);
    uint64_t vkflame_buf_address(VKFBuffer *buf);

    int vkflame_memcpy_h2d(VKFBuffer *dst, const void *src, size_t size, size_t offset);
    int vkflame_memcpy_d2h(void *dst, VKFBuffer *src, size_t size, size_t offset);
    int vkflame_memcpy_d2d(VKFBuffer *dst, VKFBuffer *src, size_t size);
    int vkflame_memset(VKFBuffer *buf, int value, size_t size);

#ifdef __cplusplus
} // extern "C"
#endif
