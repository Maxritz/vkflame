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
        bool is_wrapped;       // true = imported from HIP; free() skips vkFreeMemory
        size_t current_offset; // byte offset within this buffer for sub-allocations
                               // set by vkflame_buf_from_ptr on a range-hit
    };

    VKFBuffer *vkflame_alloc(size_t size);
    void vkflame_free(VKFBuffer *buf);
    VKFBuffer *vkflame_buf_from_ptr(void *ptr);
    uint64_t vkflame_buf_address(VKFBuffer *buf);

#ifdef _WIN32
    // Zero-copy: wrap an existing HIP device pointer in a VkBuffer via
    // VK_KHR_external_memory_win32 + hipMemGetExportHandle.
    // Returns NULL if the extension is unavailable or handle export fails
    // (caller should fall back to staged path).
    VKFBuffer *vkflame_wrap_hip_ptr(void *hip_ptr, size_t size);

    // Allocate HIP device memory that is guaranteed to be exportable
    // (hipExtMallocWithFlags with hipDeviceMallocDefault — a top-level
    // WDDM resource, not a caching-allocator sub-allocation).
    // Returns a real HIP device pointer, or nullptr on failure.
    void *vkflame_hip_alloc_exportable(size_t size);
    void vkflame_hip_free_exportable(void *ptr);

    // PyTorch CUDAPluggableAllocator hooks.
    // Signature matches what torch.cuda.memory.CUDAPluggableAllocator expects.
    // Allocates exportable HIP memory + creates zero-copy VkBuffer over it.
    void *vkflame_pytorch_malloc(size_t size, int device, void *stream);
    void vkflame_pytorch_free(void *ptr, size_t size, int device, void *stream);

    // Look up the Vulkan device address for a HIP data_ptr().
    // Returns 0 if not registered (caller should stage).
    uint64_t vkflame_buf_from_hip_ptr(uint64_t hip_ptr);
#endif

    int vkflame_memcpy_h2d(VKFBuffer *dst, const void *src, size_t size, size_t offset);
    int vkflame_memcpy_d2h(void *dst, VKFBuffer *src, size_t size, size_t offset);
    int vkflame_memcpy_d2d(VKFBuffer *dst, VKFBuffer *src, size_t size);
    int vkflame_memset(VKFBuffer *buf, int value, size_t size);

#ifdef __cplusplus
} // extern "C"
#endif
