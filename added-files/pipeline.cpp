/*
 * runtime/pipeline.cpp
 * SPIR-V pipeline cache — load, create, cache to disk.
 *
 * Key rule: push constant range size MUST exactly match the shader's
 * layout(push_constant) block size. A range that is too small causes
 * VUID-VkComputePipelineCreateInfo-layout-10069 and GPU_DEVICE_LOST.
 *
 * Verified sizes — do not change without changing the matching GLSL:
 *
 *   linear_int8       M(4) N(4) K(4) act_scale(4) activation(4)  = 20
 *   linear_fp16       M(4) N(4) K(4) activation(4)               = 16
 *   linear_coop       M(4) N(4) K(4) activation(4)               = 16
 *   flash_attention   B(4) Hq(4) Hkv(4) Sq(4) Skv(4) D(4)
 *                     scale(4) is_causal(4)                       = 32
 *   winograd_f23      OC(4) IC(4) H(4) W(4)                      = 16
 *   winograd_f45      OC(4) IC(4) H(4) W(4)                      = 16
 *   conv_direct       OC(4) IC(4) H(4) W(4) KH(4) KW(4)         = 24
 *   rms_norm          M(4) N(4) eps(4)                            = 12
 *   softmax_online    M(4) N(4)                                   = 8
 *   elementwise       N(4) op(4)                                  = 8
 *   reduce            M(4) N(4) op(4)                             = 12
 *   topk              M(4) N(4) K(4) largest(4)                  = 16
 *   sort_radix        M(4) N(4)                                   = 8
 *   embedding         V(4) D(4) B(4)                              = 12
 *   kvcache_update    seq_pos(4) n_heads(4) head_dim(4) max_seq(4) = 16
 */

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  define VK_USE_PLATFORM_WIN32_KHR
#  include <windows.h>
#  include <shlobj.h>   /* SHGetFolderPathA */
#endif

#include "device.h"

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <filesystem>
#include <fstream>
#include <vector>

/* ------------------------------------------------------------------ */
/* Push constant sizes — one entry per VKFKernelID, in order           */
/* ------------------------------------------------------------------ */

static const uint32_t k_pc_sizes[VKF_KERNEL_COUNT] = {
    /* VKF_KERNEL_LINEAR_INT8      */ 20,
    /* VKF_KERNEL_LINEAR_FP16      */ 16,
    /* VKF_KERNEL_LINEAR_COOP      */ 16,
    /* VKF_KERNEL_FLASH_ATTENTION  */ 32,
    /* VKF_KERNEL_WINOGRAD_F23     */ 16,
    /* VKF_KERNEL_WINOGRAD_F45     */ 16,
    /* VKF_KERNEL_CONV_DIRECT      */ 24,
    /* VKF_KERNEL_RMS_NORM         */ 12,
    /* VKF_KERNEL_SOFTMAX_ONLINE   */  8,
    /* VKF_KERNEL_ELEMENTWISE      */  8,
    /* VKF_KERNEL_REDUCE           */ 12,
    /* VKF_KERNEL_TOPK             */ 16,
    /* VKF_KERNEL_SORT_RADIX       */  8,
    /* VKF_KERNEL_EMBEDDING        */ 12,
    /* VKF_KERNEL_KVCACHE_UPDATE   */ 16,
};

/* Kernel name must match the key in vkf_spirv_table[] exactly */
static const char* k_kernel_names[VKF_KERNEL_COUNT] = {
    "linear_int8",
    "linear_fp16",
    "linear_coop",
    "flash_attention",
    "winograd_f23",
    "winograd_f45",
    "conv_direct",
    "rms_norm",
    "softmax_online",
    "elementwise",
    "reduce",
    "topk",
    "sort_radix",
    "embedding",
    "kvcache_update",
};

/* ------------------------------------------------------------------ */
/* Descriptor set layout — all shaders use the same pattern:           */
/*   binding 0: input A (readonly storage buffer)                      */
/*   binding 1: input B (readonly storage buffer)                      */
/*   binding 2: output  (storage buffer)                               */
/* Shaders that need fewer bindings just ignore the unused ones.       */
/* ------------------------------------------------------------------ */

#define MAX_BINDINGS 3

/* ------------------------------------------------------------------ */
/* SPIR-V table (provided by spirv_embed.cpp at link time)             */
/* ------------------------------------------------------------------ */

struct VKFSpvEntry { const char* name; const uint8_t* data; uint32_t len; };
extern "C" const VKFSpvEntry vkf_spirv_table[];

static const uint8_t* find_spirv(const char* name, uint32_t* out_len)
{
    for (int i = 0; vkf_spirv_table[i].name != nullptr; i++) {
        if (strcmp(vkf_spirv_table[i].name, name) == 0) {
            *out_len = vkf_spirv_table[i].len;
            return vkf_spirv_table[i].data;
        }
    }
    return nullptr;
}

/* ------------------------------------------------------------------ */
/* Pipeline cache file path                                             */
/* ------------------------------------------------------------------ */

static std::string cache_file_path()
{
#ifdef _WIN32
    char appdata[MAX_PATH];
    SHGetFolderPathA(nullptr, CSIDL_LOCAL_APPDATA, nullptr, 0, appdata);
    return std::string(appdata) + "\\vkflame\\pipeline_cache.bin";
#else
    const char* home = getenv("HOME");
    return std::string(home ? home : "/tmp") + "/.cache/vkflame/pipeline_cache.bin";
#endif
}

/* ------------------------------------------------------------------ */
/* Global pipeline table                                                */
/* ------------------------------------------------------------------ */

static VKFPipeline g_pipelines[VKF_KERNEL_COUNT]{};
static VkPipelineCache g_vk_cache = VK_NULL_HANDLE;

/* ------------------------------------------------------------------ */
/* Build one pipeline                                                   */
/* ------------------------------------------------------------------ */

static bool build_pipeline(VkDevice device,
                            int      kernel_id,
                            VkPipelineCache vk_cache)
{
    VKFPipeline* p = &g_pipelines[kernel_id];
    const char*  name = k_kernel_names[kernel_id];
    uint32_t     pc_size = k_pc_sizes[kernel_id];

    /* Load SPIR-V */
    uint32_t spv_len = 0;
    const uint8_t* spv = find_spirv(name, &spv_len);
    if (!spv) {
        fprintf(stderr, "[vkflame] SPIR-V not found for kernel: %s\n", name);
        return false;
    }

    /* Shader module */
    VkShaderModuleCreateInfo sm_ci{};
    sm_ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spv_len;
    sm_ci.pCode    = reinterpret_cast<const uint32_t*>(spv);

    VkShaderModule shader_module;
    if (vkCreateShaderModule(device, &sm_ci, nullptr, &shader_module) != VK_SUCCESS) {
        fprintf(stderr, "[vkflame] vkCreateShaderModule failed for: %s\n", name);
        return false;
    }

    /* Descriptor set layout — 3 storage buffer bindings */
    VkDescriptorSetLayoutBinding bindings[MAX_BINDINGS]{};
    for (int i = 0; i < MAX_BINDINGS; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo ds_ci{};
    ds_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds_ci.bindingCount = MAX_BINDINGS;
    ds_ci.pBindings    = bindings;

    if (vkCreateDescriptorSetLayout(device, &ds_ci, nullptr, &p->ds_layout) != VK_SUCCESS) {
        vkDestroyShaderModule(device, shader_module, nullptr);
        return false;
    }

    /* Pipeline layout — push constants sized exactly to shader's PC block */
    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset     = 0;
    pc_range.size       = pc_size;   /* MUST match layout(push_constant) block in shader */

    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount         = 1;
    pl_ci.pSetLayouts            = &p->ds_layout;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges    = &pc_range;

    if (vkCreatePipelineLayout(device, &pl_ci, nullptr, &p->layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, p->ds_layout, nullptr);
        vkDestroyShaderModule(device, shader_module, nullptr);
        return false;
    }

    /* Compute pipeline */
    VkComputePipelineCreateInfo cp_ci{};
    cp_ci.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp_ci.stage.sType        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp_ci.stage.stage        = VK_SHADER_STAGE_COMPUTE_BIT;
    cp_ci.stage.module       = shader_module;
    cp_ci.stage.pName        = "main";
    cp_ci.layout             = p->layout;

    VkResult rc = vkCreateComputePipelines(device, vk_cache, 1, &cp_ci, nullptr, &p->pipeline);

    vkDestroyShaderModule(device, shader_module, nullptr);  /* always free — pipeline holds a ref */

    if (rc != VK_SUCCESS) {
        fprintf(stderr, "[vkflame] vkCreateComputePipelines failed for %s: %d\n", name, (int)rc);
        vkDestroyPipelineLayout(device, p->layout, nullptr);
        vkDestroyDescriptorSetLayout(device, p->ds_layout, nullptr);
        return false;
    }

    p->push_constant_size = pc_size;
    return true;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

VKF_API int vkflame_pipelines_init()
{
    VKFContext* ctx = vkflame_get_context();
    if (!ctx) return -1;

    VkDevice device = ctx->device;

    /* Load pipeline cache from disk */
    std::string cache_path = cache_file_path();
    std::vector<uint8_t> cache_data;

    {
        std::ifstream f(cache_path, std::ios::binary | std::ios::ate);
        if (f.is_open()) {
            size_t sz = f.tellg();
            f.seekg(0);
            cache_data.resize(sz);
            f.read(reinterpret_cast<char*>(cache_data.data()), sz);
        }
    }

    VkPipelineCacheCreateInfo cache_ci{};
    cache_ci.sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cache_ci.initialDataSize = cache_data.size();
    cache_ci.pInitialData    = cache_data.empty() ? nullptr : cache_data.data();
    vkCreatePipelineCache(device, &cache_ci, nullptr, &g_vk_cache);

    /* Build all pipelines */
    int failed = 0;
    for (int i = 0; i < VKF_KERNEL_COUNT; i++) {
        if (!build_pipeline(device, i, g_vk_cache)) {
            fprintf(stderr, "[vkflame] failed to build pipeline: %s\n", k_kernel_names[i]);
            failed++;
        }
    }

    /* Save updated cache to disk */
    if (g_vk_cache != VK_NULL_HANDLE) {
        size_t data_size = 0;
        vkGetPipelineCacheData(device, g_vk_cache, &data_size, nullptr);
        if (data_size > 0) {
            std::vector<uint8_t> data(data_size);
            vkGetPipelineCacheData(device, g_vk_cache, &data_size, data.data());
            std::filesystem::create_directories(
                std::filesystem::path(cache_path).parent_path());
            std::ofstream f(cache_path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(data.data()), data_size);
        }
    }

    return failed > 0 ? -1 : 0;
}

VKF_API VKFPipeline* vkflame_get_pipeline(VKFKernelID id)
{
    if (id < 0 || id >= VKF_KERNEL_COUNT) return nullptr;
    if (!g_pipelines[id].pipeline) return nullptr;
    return &g_pipelines[id];
}

VKF_API void vkflame_pipelines_destroy()
{
    VKFContext* ctx = vkflame_get_context();
    if (!ctx) return;
    VkDevice device = ctx->device;

    for (int i = 0; i < VKF_KERNEL_COUNT; i++) {
        auto& p = g_pipelines[i];
        if (p.pipeline)  vkDestroyPipeline(device, p.pipeline, nullptr);
        if (p.layout)    vkDestroyPipelineLayout(device, p.layout, nullptr);
        if (p.ds_layout) vkDestroyDescriptorSetLayout(device, p.ds_layout, nullptr);
        p = {};
    }

    if (g_vk_cache != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(device, g_vk_cache, nullptr);
        g_vk_cache = VK_NULL_HANDLE;
    }
}
