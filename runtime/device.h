#pragma once
#include <vulkan/vulkan.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    struct VKFFeatures
    {
        bool has_cooperative_matrix;
        bool has_float8;
        bool has_integer_dot_product;
        bool has_float16;
        bool has_int8;
        bool has_buffer_device_address;
        uint32_t subgroup_size;
        uint32_t max_workgroup_x;
        uint32_t max_compute_shared_memory;
        char device_name[256];
        uint32_t vendor_id; // 0x1002 AMD, 0x10DE NVIDIA, 0x8086 Intel
    };

    struct VKFContext
    {
        VkInstance instance;
        VkPhysicalDevice physical_device;
        VkDevice device;
        VkQueue compute_queue;
        VkQueue transfer_queue;
        uint32_t compute_family;
        uint32_t transfer_family;
        VkCommandPool command_pool;
        VKFFeatures features;
    };

    VKFContext *vkflame_get_context();
    int vkflame_init();
    void vkflame_shutdown();
    void vkflame_print_info();

#ifdef __cplusplus
} // extern "C"
#endif
