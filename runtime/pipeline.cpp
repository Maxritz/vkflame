#include "pipeline.h"
#include "device.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>

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

// ── SPIR-V table from spirv_embed.cpp ────────────────────────────
struct VKFSpvEntry
{
    const char *name;
    const uint8_t *data;
    uint32_t len;
};
extern "C" const VKFSpvEntry vkf_spirv_table[];

// ── Per-kernel descriptor layout config ─────────────────────────
struct KernelConfig
{
    const char *spv_name;        // matches spirv_embed entry
    uint32_t push_constant_size; // bytes
    uint32_t n_bindings;         // number of storage buffer descriptors
};

// Binding counts per kernel (storage buffers only, set 0)
static const KernelConfig k_configs[VKF_KERNEL_COUNT] = {
    {"linear_int8", 20, 5},         // 0: A,B,C,D,scale
    {"linear_fp16", 16, 4},         // 1: A,B,C,D
    {"linear_coop", 16, 4},         // 2: A,B,C,D
    {"flash_attention", 32, 4},     // 3: Q,K,V,O
    {"winograd_f23", 28, 3},        // 4: input,filter,output  (7 × uint = 28 bytes)
    {"winograd_f23", 28, 3},        // 5: (same shader, different config)
    {"linear_fp16", 16, 4},         // 6: conv_direct fallback
    {"rms_norm", 12, 3},            // 7: X,gamma,Y
    {"softmax_online", 8, 2},       // 8: X,Y
    {"linear_fp16", 16, 4},         // 9: elementwise (use linear)
    {"softmax_online", 8, 2},       // 10: reduce
    {"topk", 16, 3},                // 11: X,values,indices
    {"topk", 16, 3},                // 12: sort (reuse topk)
    {"embedding", 12, 3},           // 13: weight,indices,out
    {"kvcache_update", 16, 2},      // 14: cache,new_kv
    {"linear_fp16_fp32out", 16, 4}, // 15: A(fp16),B(fp16),C(fp32 unused),D(fp32)
    // ── ggml dequantization kernels ──────────────────────────────────
    {"dequant_q4_0", 4, 2}, // 16: src(Q4_0 packed), dst(float)
    {"dequant_q4_1", 4, 2}, // 17: src(Q4_1 packed), dst(float)
    {"dequant_q8_0", 4, 2}, // 18: src(Q8_0 packed), dst(float)
    {"dequant_q5_0", 4, 2}, // 19: src(Q5_0 packed), dst(float)
    {"dequant_q4_k", 4, 2}, // 20: src(Q4_K packed), dst(float)
    {"dequant_q5_k", 4, 2}, // 21: src(Q5_K packed), dst(float)
    {"dequant_q6_k", 4, 2}, // 22: src(Q6_K packed), dst(float)
    // ── ggml element-wise + utility kernels ──────────────────────────
    {"elementwise_f32", 8, 2}, // 23: src,dst  + {n,op}
    {"rope_neox", 32, 3},      // 24: src,pos,dst + {ne0,ne1,n_dims,n_seqs,theta_scale,freq_scale,p0,p1}
    {"scale_f32", 12, 2},      // 25: src,dst + {n,scale,bias}
    {"binop_f32", 16, 3},      // 26: src0,src1,dst + {n,op,src1_n,pad}
    {"rms_norm_f32", 12, 2},   // 27: src,dst + {M,N,eps}
};

static VKFPipeline g_pipelines[VKF_KERNEL_COUNT] = {};
static bool g_pipelines_ready = false;

static std::string get_cache_path()
{
    const char *home = getenv("USERPROFILE");
    if (!home)
        home = getenv("HOME");
    if (!home)
        home = ".";
    return std::string(home) + "/.cache/vkflame/pipeline_cache.bin";
}

static const uint8_t *find_spv(const char *name, uint32_t *out_len)
{
    for (int i = 0; vkf_spirv_table[i].name != nullptr; i++)
    {
        if (strcmp(vkf_spirv_table[i].name, name) == 0)
        {
            *out_len = vkf_spirv_table[i].len;
            return vkf_spirv_table[i].data;
        }
    }
    return nullptr;
}

static VkPipeline create_pipeline(VkDevice device, VkPipelineLayout layout,
                                  const uint8_t *spv, uint32_t spv_len,
                                  VkPipelineCache cache)
{
    VkShaderModuleCreateInfo sm_ci = {};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spv_len;
    sm_ci.pCode = (const uint32_t *)spv;

    VkShaderModule shader;
    VKF_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &shader));

    VkComputePipelineCreateInfo pipe_ci = {};
    pipe_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipe_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipe_ci.stage.module = shader;
    pipe_ci.stage.pName = "main";
    pipe_ci.layout = layout;

    VkPipeline pipeline;
    VKF_CHECK(vkCreateComputePipelines(device, cache, 1, &pipe_ci, nullptr, &pipeline));
    vkDestroyShaderModule(device, shader, nullptr);
    return pipeline;
}

extern "C"
{

    int vkflame_pipelines_init()
    {
        if (g_pipelines_ready)
            return 0;
        VKFContext *ctx = vkflame_get_context();
        if (!ctx)
            return -1;

        // ── Load pipeline cache ──────────────────────────────────────
        std::string cache_path = get_cache_path();
        std::vector<uint8_t> cache_data;
        {
            std::ifstream f(cache_path, std::ios::binary);
            if (f)
            {
                f.seekg(0, std::ios::end);
                size_t sz = f.tellg();
                f.seekg(0);
                cache_data.resize(sz);
                f.read((char *)cache_data.data(), sz);
            }
        }

        VkPipelineCacheCreateInfo pcc = {};
        pcc.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        pcc.initialDataSize = cache_data.size();
        pcc.pInitialData = cache_data.empty() ? nullptr : cache_data.data();

        VkPipelineCache pipeline_cache;
        VKF_CHECK(vkCreatePipelineCache(ctx->device, &pcc, nullptr, &pipeline_cache));

        // ── Create pipelines ─────────────────────────────────────────────
        for (int i = 0; i < VKF_KERNEL_COUNT; i++)
        {
            const auto &cfg = k_configs[i];
            uint32_t n_bind = cfg.n_bindings;

            // Descriptor set layout — all storage buffers
            std::vector<VkDescriptorSetLayoutBinding> bindings(n_bind);
            for (uint32_t b = 0; b < n_bind; b++)
            {
                bindings[b] = {};
                bindings[b].binding = b;
                bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[b].descriptorCount = 1;
                bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }

            VkDescriptorSetLayoutCreateInfo dsl_ci = {};
            dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dsl_ci.bindingCount = n_bind;
            dsl_ci.pBindings = bindings.data();

            VKF_CHECK(vkCreateDescriptorSetLayout(ctx->device, &dsl_ci, nullptr,
                                                  &g_pipelines[i].ds_layout));

            VkPushConstantRange pc_range = {};
            pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pc_range.offset = 0;
            pc_range.size = cfg.push_constant_size ? cfg.push_constant_size : 4;

            VkPipelineLayoutCreateInfo pl_ci = {};
            pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_ci.setLayoutCount = 1;
            pl_ci.pSetLayouts = &g_pipelines[i].ds_layout;
            pl_ci.pushConstantRangeCount = 1;
            pl_ci.pPushConstantRanges = &pc_range;

            VKF_CHECK(vkCreatePipelineLayout(ctx->device, &pl_ci, nullptr,
                                             &g_pipelines[i].layout));

            g_pipelines[i].push_constant_size = pc_range.size;

            // Find SPIR-V
            uint32_t spv_len = 0;
            const uint8_t *spv = find_spv(cfg.spv_name, &spv_len);
            if (!spv)
            {
                fprintf(stderr, "[vkflame] SPIR-V not found for kernel[%d]: %s\n",
                        i, cfg.spv_name);
                // Create a dummy no-op pipeline won't succeed, skip
                g_pipelines[i].pipeline = VK_NULL_HANDLE;
                continue;
            }

            g_pipelines[i].pipeline = create_pipeline(ctx->device, g_pipelines[i].layout,
                                                      spv, spv_len, pipeline_cache);
        }

        // ── Save pipeline cache ──────────────────────────────────────
        {
            std::filesystem::path cp(cache_path);
            std::filesystem::create_directories(cp.parent_path());
            size_t data_size = 0;
            vkGetPipelineCacheData(ctx->device, pipeline_cache, &data_size, nullptr);
            std::vector<uint8_t> save_data(data_size);
            vkGetPipelineCacheData(ctx->device, pipeline_cache, &data_size, save_data.data());
            std::ofstream f(cache_path, std::ios::binary);
            if (f)
                f.write((const char *)save_data.data(), data_size);
        }

        vkDestroyPipelineCache(ctx->device, pipeline_cache, nullptr);
        g_pipelines_ready = true;
        return 0;
    }

    VKFPipeline *vkflame_get_pipeline(VKFKernelID id)
    {
        if (!g_pipelines_ready || id >= VKF_KERNEL_COUNT)
            return nullptr;
        return &g_pipelines[id];
    }

    void vkflame_pipelines_destroy()
    {
        VKFContext *ctx = vkflame_get_context();
        if (!ctx)
            return;
        for (int i = 0; i < VKF_KERNEL_COUNT; i++)
        {
            if (g_pipelines[i].pipeline)
                vkDestroyPipeline(ctx->device, g_pipelines[i].pipeline, nullptr);
            if (g_pipelines[i].layout)
                vkDestroyPipelineLayout(ctx->device, g_pipelines[i].layout, nullptr);
            if (g_pipelines[i].ds_layout)
                vkDestroyDescriptorSetLayout(ctx->device, g_pipelines[i].ds_layout, nullptr);
        }
        g_pipelines_ready = false;
    }

} // extern "C"
