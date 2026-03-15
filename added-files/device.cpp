/*
 * runtime/device.cpp
 * Vulkan device context — init, feature detection, queue setup.
 *
 * Key rule: use VkPhysicalDeviceVulkan12Features for ALL 1.2 features.
 * Do NOT chain individual extension structs alongside it — that violates
 * VUID-VkDeviceCreateInfo-pNext-02830 and crashes the validation layer.
 */

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  define VK_USE_PLATFORM_WIN32_KHR
#  include <windows.h>
#endif

#include "device.h"

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

#define VKF_CHECK(call)                                                 \
    do {                                                                \
        VkResult _r = (call);                                           \
        if (_r != VK_SUCCESS) {                                         \
            fprintf(stderr, "[vkflame] VkResult=%d at %s:%d\n",        \
                    (int)_r, __FILE__, __LINE__);                       \
        }                                                               \
    } while (0)

static VKFContext* g_ctx = nullptr;

/* ------------------------------------------------------------------ */
/* Instance creation                                                    */
/* ------------------------------------------------------------------ */

static VkInstance create_instance(bool validation)
{
    const char* layers[]     = { "VK_LAYER_KHRONOS_validation" };
    const char* extensions[] = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };

    VkApplicationInfo app_info{};
    app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName   = "vkflame";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion         = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &app_info;
    ci.enabledExtensionCount   = 1;
    ci.ppEnabledExtensionNames = extensions;

    if (validation) {
        ci.enabledLayerCount   = 1;
        ci.ppEnabledLayerNames = layers;
    }

    VkInstance instance;
    VKF_CHECK(vkCreateInstance(&ci, nullptr, &instance));
    return instance;
}

/* ------------------------------------------------------------------ */
/* Physical device selection                                            */
/* ------------------------------------------------------------------ */

static VkPhysicalDevice pick_physical_device(VkInstance instance)
{
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) return VK_NULL_HANDLE;

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    VkPhysicalDevice best       = VK_NULL_HANDLE;
    VkDeviceSize     best_vram  = 0;

    for (auto pd : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);

        /* Prefer discrete GPU */
        bool discrete = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);

        /* Get VRAM size from memory heaps */
        VkPhysicalDeviceMemoryProperties mem;
        vkGetPhysicalDeviceMemoryProperties(pd, &mem);
        VkDeviceSize vram = 0;
        for (uint32_t i = 0; i < mem.memoryHeapCount; i++) {
            if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                vram += mem.memoryHeaps[i].size;
        }

        if (best == VK_NULL_HANDLE || (discrete && vram > best_vram)) {
            best      = pd;
            best_vram = vram;
        }
    }

    return best;
}

/* ------------------------------------------------------------------ */
/* Feature detection                                                    */
/* ------------------------------------------------------------------ */

static void detect_features(VkPhysicalDevice pd, VKFFeatures* f)
{
    /* Core properties */
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    strncpy(f->device_name, props.deviceName, 255);
    f->device_name[255]      = '\0';
    f->vendor_id             = props.vendorID;
    f->max_workgroup_x       = props.limits.maxComputeWorkGroupSize[0];
    f->max_compute_shared_memory = props.limits.maxComputeSharedMemorySize;

    /* Subgroup size — query actual execution size, not max */
    VkPhysicalDeviceVulkan11Properties vk11props{};
    vk11props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &vk11props;
    vkGetPhysicalDeviceProperties2(pd, &props2);
    f->subgroup_size = vk11props.subgroupSize;

    /* Feature availability — chain only structs that don't overlap Vk12 */
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{};
    coop.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;

    VkPhysicalDeviceVulkan12Features vk12{};
    vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12.pNext = &coop;

    VkPhysicalDeviceFeatures2 feat2{};
    feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feat2.pNext = &vk12;
    vkGetPhysicalDeviceFeatures2(pd, &feat2);

    f->has_float16              = (vk12.shaderFloat16 == VK_TRUE);
    f->has_int8                 = (vk12.shaderInt8    == VK_TRUE);
    f->has_buffer_device_address = (vk12.bufferDeviceAddress == VK_TRUE);
    f->has_integer_dot_product  = VK_FALSE; /* checked via extension below */
    f->has_cooperative_matrix   = (coop.cooperativeMatrix == VK_TRUE);
    f->has_float8               = VK_FALSE; /* not widely supported yet */

    /* Check integer dot product extension */
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, exts.data());
    for (auto& e : exts) {
        if (strcmp(e.extensionName, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) == 0)
            f->has_integer_dot_product = VK_TRUE;
        if (strcmp(e.extensionName, "VK_KHR_shader_float8") == 0)  /* provisional */
            f->has_float8 = VK_TRUE;
    }
}

/* ------------------------------------------------------------------ */
/* Logical device creation                                              */
/* ------------------------------------------------------------------ */

static VkDevice create_device(VkPhysicalDevice pd,
                               uint32_t compute_family,
                               uint32_t transfer_family,
                               bool     same_family)
{
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_cis[2]{};
    uint32_t queue_count = 1;

    queue_cis[0].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_cis[0].queueFamilyIndex = compute_family;
    queue_cis[0].queueCount       = 1;
    queue_cis[0].pQueuePriorities = &priority;

    if (!same_family) {
        queue_cis[1].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_cis[1].queueFamilyIndex = transfer_family;
        queue_cis[1].queueCount       = 1;
        queue_cis[1].pQueuePriorities = &priority;
        queue_count = 2;
    }

    /*
     * ALL Vulkan 1.2 features go here — and ONLY here.
     * Do not also chain VkPhysicalDeviceShaderFloat16Int8Features,
     * VkPhysicalDeviceBufferDeviceAddressFeatures, or
     * VkPhysicalDevice8BitStorageFeatures alongside this struct.
     * That violates VUID-VkDeviceCreateInfo-pNext-02830.
     */
    VkPhysicalDeviceVulkan12Features vk12{};
    vk12.sType                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12.shaderFloat16            = VK_TRUE;
    vk12.shaderInt8               = VK_TRUE;
    vk12.storageBuffer8BitAccess  = VK_TRUE;  /* required by linear_int8.glsl */
    vk12.bufferDeviceAddress      = VK_TRUE;
    vk12.pNext                    = nullptr;  /* nothing else — see note above */

    /*
     * Vulkan 1.3 features.
     * Chain after vk12.
     */
    VkPhysicalDeviceVulkan13Features vk13{};
    vk13.sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vk13.synchronization2 = VK_TRUE;
    vk13.dynamicRendering = VK_FALSE;
    vk13.pNext            = nullptr;
    vk12.pNext            = &vk13;

    /* Required device extensions */
    std::vector<const char*> exts = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,        /* needed on some drivers even for compute */
    };

    /* Optional — add only if supported */
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> avail(ext_count);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, avail.data());
    auto has_ext = [&](const char* name) {
        for (auto& e : avail)
            if (strcmp(e.extensionName, name) == 0) return true;
        return false;
    };

    if (has_ext(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
        exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

    VkDeviceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.pNext                   = &vk12;
    ci.queueCreateInfoCount    = queue_count;
    ci.pQueueCreateInfos       = queue_cis;
    ci.enabledExtensionCount   = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();

    VkDevice device;
    VKF_CHECK(vkCreateDevice(pd, &ci, nullptr, &device));
    return device;
}

/* ------------------------------------------------------------------ */
/* Queue family selection                                               */
/* ------------------------------------------------------------------ */

static void find_queue_families(VkPhysicalDevice pd,
                                uint32_t* compute,
                                uint32_t* transfer,
                                bool*     same)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, families.data());

    *compute  = UINT32_MAX;
    *transfer = UINT32_MAX;

    /* Find a compute-only family first (no graphics) — dedicated compute */
    for (uint32_t i = 0; i < count; i++) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            if (*compute == UINT32_MAX) *compute = i;
        }
    }
    /* Fall back to any compute family */
    if (*compute == UINT32_MAX) {
        for (uint32_t i = 0; i < count; i++) {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                *compute = i;
                break;
            }
        }
    }

    /* Find a transfer-only family */
    for (uint32_t i = 0; i < count; i++) {
        if ((families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            *transfer = i;
            break;
        }
    }
    if (*transfer == UINT32_MAX) *transfer = *compute;

    *same = (*compute == *transfer);
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

VKF_API VKFContext* vkflame_get_context() { return g_ctx; }

VKF_API int vkflame_init()
{
    if (g_ctx) return 0;  /* already initialised */

    g_ctx = new VKFContext{};

#ifdef NDEBUG
    bool validation = false;
#else
    bool validation = true;
#endif

    g_ctx->instance = create_instance(validation);
    if (!g_ctx->instance) { delete g_ctx; g_ctx = nullptr; return -1; }

    g_ctx->physical_device = pick_physical_device(g_ctx->instance);
    if (!g_ctx->physical_device) {
        fprintf(stderr, "[vkflame] no Vulkan GPU found\n");
        delete g_ctx; g_ctx = nullptr; return -2;
    }

    detect_features(g_ctx->physical_device, &g_ctx->features);

    find_queue_families(g_ctx->physical_device,
                        &g_ctx->compute_family,
                        &g_ctx->transfer_family,
                        &(bool){false});

    bool same = (g_ctx->compute_family == g_ctx->transfer_family);
    g_ctx->device = create_device(g_ctx->physical_device,
                                  g_ctx->compute_family,
                                  g_ctx->transfer_family,
                                  same);
    if (!g_ctx->device) { delete g_ctx; g_ctx = nullptr; return -3; }

    vkGetDeviceQueue(g_ctx->device, g_ctx->compute_family,  0, &g_ctx->compute_queue);
    vkGetDeviceQueue(g_ctx->device, g_ctx->transfer_family, 0, &g_ctx->transfer_queue);

    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.queueFamilyIndex = g_ctx->compute_family;
    pool_ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VKF_CHECK(vkCreateCommandPool(g_ctx->device, &pool_ci, nullptr, &g_ctx->command_pool));

    /* Init pipelines after device is ready */
    int rc = vkflame_pipelines_init();
    if (rc != 0) { vkflame_shutdown(); return rc; }

    vkflame_print_info();
    return 0;
}

VKF_API void vkflame_shutdown()
{
    if (!g_ctx) return;
    vkflame_pipelines_destroy();
    if (g_ctx->command_pool) vkDestroyCommandPool(g_ctx->device, g_ctx->command_pool, nullptr);
    if (g_ctx->device)       vkDestroyDevice(g_ctx->device, nullptr);
    if (g_ctx->instance)     vkDestroyInstance(g_ctx->instance, nullptr);
    delete g_ctx;
    g_ctx = nullptr;
}

VKF_API void vkflame_print_info()
{
    if (!g_ctx) return;
    auto& f = g_ctx->features;
    fprintf(stderr,
        "[vkflame] %s  subgroup:%u  coop_matrix:%s  fp8:%s  int_dot:%s  fp16:%s  int8:%s\n",
        f.device_name,
        f.subgroup_size,
        f.has_cooperative_matrix   ? "yes" : "no",
        f.has_float8               ? "yes" : "no",
        f.has_integer_dot_product  ? "yes" : "no",
        f.has_float16              ? "yes" : "no",
        f.has_int8                 ? "yes" : "no");
}
