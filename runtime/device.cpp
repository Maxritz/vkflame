#include "device.h"
#include "pipeline.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

// ── VKF_CHECK macro ──────────────────────────────────────────────
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

static VKFContext g_ctx = {};
static bool g_initialized = false;

// ── Validation layer callback ────────────────────────────────────
#ifndef NDEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT *data,
    void *)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        fprintf(stderr, "[vkflame validation] %s\n", data->pMessage);
    return VK_FALSE;
}
#endif

// ── Pick best physical device ────────────────────────────────────
static VkPhysicalDevice pick_device(VkInstance instance)
{
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0)
        return VK_NULL_HANDLE;

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    VkPhysicalDevice best = VK_NULL_HANDLE;
    VkDeviceSize best_vram = 0;
    bool best_discrete = false;

    for (auto dev : devices)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);

        VkPhysicalDeviceMemoryProperties mem;
        vkGetPhysicalDeviceMemoryProperties(dev, &mem);

        VkDeviceSize vram = 0;
        for (uint32_t i = 0; i < mem.memoryHeapCount; i++)
        {
            if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                vram += mem.memoryHeaps[i].size;
        }

        bool discrete = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
        // Prefer discrete; break ties by VRAM
        if (best == VK_NULL_HANDLE ||
            (discrete && !best_discrete) ||
            (discrete == best_discrete && vram > best_vram))
        {
            best = dev;
            best_vram = vram;
            best_discrete = discrete;
        }
    }
    return best;
}

// ── Queue family selection ───────────────────────────────────────
static uint32_t find_queue_family(VkPhysicalDevice dev, VkQueueFlags required, VkQueueFlags excluded)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> fams(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, fams.data());
    for (uint32_t i = 0; i < count; i++)
    {
        if ((fams[i].queueFlags & required) == required &&
            (fams[i].queueFlags & excluded) == 0)
            return i;
    }
    return UINT32_MAX;
}

static uint32_t find_queue_family_any(VkPhysicalDevice dev, VkQueueFlags required)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> fams(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, fams.data());
    for (uint32_t i = 0; i < count; i++)
    {
        if ((fams[i].queueFlags & required) == required)
            return i;
    }
    return UINT32_MAX;
}

// ── Check extension support ──────────────────────────────────────
static bool has_device_extension(VkPhysicalDevice dev, const char *name)
{
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, exts.data());
    for (auto &e : exts)
        if (strcmp(e.extensionName, name) == 0)
            return true;
    return false;
}

extern "C"
{

    int vkflame_init()
    {
        if (g_initialized)
            return 0;

        // ── Instance ─────────────────────────────────────────────────
        VkApplicationInfo app_info = {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "vkflame";
        app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        app_info.apiVersion = VK_API_VERSION_1_4;

        std::vector<const char *> instance_layers;
        std::vector<const char *> instance_exts;

#ifndef NDEBUG
        instance_layers.push_back("VK_LAYER_KHRONOS_validation");
        instance_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

        VkInstanceCreateInfo inst_ci = {};
        inst_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        inst_ci.pApplicationInfo = &app_info;
        inst_ci.enabledLayerCount = (uint32_t)instance_layers.size();
        inst_ci.ppEnabledLayerNames = instance_layers.data();
        inst_ci.enabledExtensionCount = (uint32_t)instance_exts.size();
        inst_ci.ppEnabledExtensionNames = instance_exts.data();

        VkResult r = vkCreateInstance(&inst_ci, nullptr, &g_ctx.instance);
        if (r != VK_SUCCESS)
        {
            fprintf(stderr, "[vkflame] vkCreateInstance failed: %d\n", (int)r);
            return -1;
        }

#ifndef NDEBUG
        VkDebugUtilsMessengerCreateInfoEXT dbg_ci = {};
        dbg_ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbg_ci.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dbg_ci.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        dbg_ci.pfnUserCallback = debug_callback;
        auto create_messenger = (PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(g_ctx.instance, "vkCreateDebugUtilsMessengerEXT");
        if (create_messenger)
        {
            VkDebugUtilsMessengerEXT messenger;
            create_messenger(g_ctx.instance, &dbg_ci, nullptr, &messenger);
        }
#endif

        // ── Physical device ──────────────────────────────────────────
        g_ctx.physical_device = pick_device(g_ctx.instance);
        if (g_ctx.physical_device == VK_NULL_HANDLE)
        {
            fprintf(stderr, "[vkflame] No Vulkan GPU found\n");
            return -1;
        }

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(g_ctx.physical_device, &props);
        strncpy(g_ctx.features.device_name, props.deviceName, 255);
        g_ctx.features.vendor_id = props.vendorID;
        g_ctx.features.max_workgroup_x = props.limits.maxComputeWorkGroupSize[0];
        g_ctx.features.max_compute_shared_memory = props.limits.maxComputeSharedMemorySize;

        // ── Subgroup size ────────────────────────────────────────────
        VkPhysicalDeviceSubgroupProperties sg_props = {};
        sg_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 props2 = {};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &sg_props;
        vkGetPhysicalDeviceProperties2(g_ctx.physical_device, &props2);
        g_ctx.features.subgroup_size = sg_props.subgroupSize;

        // ── Feature detection ────────────────────────────────────────
        VkPhysicalDeviceFeatures2 feat2 = {};
        feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

        VkPhysicalDeviceShaderFloat16Int8Features f16i8 = {};
        f16i8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        feat2.pNext = &f16i8;

        VkPhysicalDeviceVulkan12Features vk12 = {};
        vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        f16i8.pNext = &vk12;

        VkPhysicalDeviceVulkan13Features vk13 = {};
        vk13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vk12.pNext = &vk13;

        vkGetPhysicalDeviceFeatures2(g_ctx.physical_device, &feat2);

        g_ctx.features.has_float16 = f16i8.shaderFloat16;
        g_ctx.features.has_int8 = f16i8.shaderInt8;
        g_ctx.features.has_buffer_device_address = vk12.bufferDeviceAddress;

        // Integer dot product (Vulkan 1.3 core)
        VkPhysicalDeviceShaderIntegerDotProductFeatures dot_feat = {};
        dot_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
        VkPhysicalDeviceFeatures2 dot_feat2 = {};
        dot_feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        dot_feat2.pNext = &dot_feat;
        vkGetPhysicalDeviceFeatures2(g_ctx.physical_device, &dot_feat2);
        g_ctx.features.has_integer_dot_product = dot_feat.shaderIntegerDotProduct;

        // Optional extensions
        g_ctx.features.has_cooperative_matrix =
            has_device_extension(g_ctx.physical_device, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        // FP8 — extension may not be present on all drivers
        g_ctx.features.has_float8 = false; // conservative default

        // ── Queue families ───────────────────────────────────────────
        // Try to find a dedicated transfer queue (no compute flag)
        uint32_t compute_fam = find_queue_family_any(g_ctx.physical_device, VK_QUEUE_COMPUTE_BIT);
        uint32_t transfer_fam = find_queue_family(g_ctx.physical_device,
                                                  VK_QUEUE_TRANSFER_BIT, VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT);
        if (transfer_fam == UINT32_MAX)
            transfer_fam = compute_fam; // fallback

        g_ctx.compute_family = compute_fam;
        g_ctx.transfer_family = transfer_fam;

        // ── Logical device ───────────────────────────────────────────
        float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queue_cis;

        VkDeviceQueueCreateInfo qci = {};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = compute_fam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        queue_cis.push_back(qci);

        if (transfer_fam != compute_fam)
        {
            VkDeviceQueueCreateInfo tqci = qci;
            tqci.queueFamilyIndex = transfer_fam;
            queue_cis.push_back(tqci);
        }

        std::vector<const char *> device_exts;
        device_exts.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        device_exts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        if (g_ctx.features.has_cooperative_matrix)
            device_exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

        // Filter to only extensions that exist
        std::vector<const char *> valid_exts;
        for (auto ext : device_exts)
        {
            if (has_device_extension(g_ctx.physical_device, ext))
                valid_exts.push_back(ext);
        }

        // Vulkan 1.1 features — storageBuffer16BitAccess required by GL_EXT_shader_16bit_storage
        VkPhysicalDeviceVulkan11Features en_vk11 = {};
        en_vk11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        en_vk11.storageBuffer16BitAccess = VK_TRUE;
        en_vk11.uniformAndStorageBuffer16BitAccess = VK_TRUE;

        // Vulkan 1.2 features — must NOT coexist with VkPhysicalDeviceShaderFloat16Int8Features
        // Move shaderFloat16/shaderInt8 here instead.
        VkPhysicalDeviceVulkan12Features en_vk12 = {};
        en_vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        en_vk12.bufferDeviceAddress = g_ctx.features.has_buffer_device_address;
        en_vk12.shaderFloat16 = g_ctx.features.has_float16;
        en_vk12.shaderInt8 = g_ctx.features.has_int8;
        en_vk12.storageBuffer8BitAccess = g_ctx.features.has_int8; // for linear_int8.glsl
        en_vk12.vulkanMemoryModel = VK_TRUE;                       // required by GL_KHR_memory_scope_semantics (linear_coop)
        en_vk12.vulkanMemoryModelDeviceScope = VK_TRUE;
        en_vk12.pNext = &en_vk11;

        VkPhysicalDeviceShaderIntegerDotProductFeatures en_dot = {};
        en_dot.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
        en_dot.shaderIntegerDotProduct = g_ctx.features.has_integer_dot_product;
        en_vk11.pNext = &en_dot;

        // CooperativeMatrix feature struct (extension, not promoted to 1.3 core on all drivers)
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR en_coop = {};
        en_coop.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        en_coop.cooperativeMatrix = g_ctx.features.has_cooperative_matrix ? VK_TRUE : VK_FALSE;
        en_dot.pNext = &en_coop;

        VkDeviceCreateInfo dev_ci = {};
        dev_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dev_ci.pNext = &en_vk12; // vk12 → vk11 → dot; no separate extension structs
        dev_ci.queueCreateInfoCount = (uint32_t)queue_cis.size();
        dev_ci.pQueueCreateInfos = queue_cis.data();
        dev_ci.enabledExtensionCount = (uint32_t)valid_exts.size();
        dev_ci.ppEnabledExtensionNames = valid_exts.data();

        r = vkCreateDevice(g_ctx.physical_device, &dev_ci, nullptr, &g_ctx.device);
        if (r != VK_SUCCESS)
        {
            fprintf(stderr, "[vkflame] vkCreateDevice failed: %d\n", (int)r);
            return -1;
        }

        vkGetDeviceQueue(g_ctx.device, compute_fam, 0, &g_ctx.compute_queue);
        vkGetDeviceQueue(g_ctx.device, transfer_fam, 0, &g_ctx.transfer_queue);

        // ── Command pool ─────────────────────────────────────────────
        VkCommandPoolCreateInfo pool_ci = {};
        pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_ci.queueFamilyIndex = compute_fam;
        VKF_CHECK(vkCreateCommandPool(g_ctx.device, &pool_ci, nullptr, &g_ctx.command_pool));

        g_initialized = true;
        vkflame_print_info();
        vkflame_pipelines_init();
        return 0;
    }

    void vkflame_shutdown()
    {
        if (!g_initialized)
            return;
        vkDestroyCommandPool(g_ctx.device, g_ctx.command_pool, nullptr);
        vkDestroyDevice(g_ctx.device, nullptr);
        vkDestroyInstance(g_ctx.instance, nullptr);
        g_initialized = false;
    }

    VKFContext *vkflame_get_context()
    {
        return g_initialized ? &g_ctx : nullptr;
    }

    void vkflame_print_info()
    {
        const auto &f = g_ctx.features;
        fprintf(stdout, "[vkflame] %s  subgroup:%u  coop_matrix:%s  fp8:%s  int_dot:%s\n",
                f.device_name,
                f.subgroup_size,
                f.has_cooperative_matrix ? "yes" : "no",
                f.has_float8 ? "yes" : "no",
                f.has_integer_dot_product ? "yes" : "no");
        fflush(stdout);
    }

} // extern "C"
