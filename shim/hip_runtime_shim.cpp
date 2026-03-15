// hip_runtime_shim.cpp — vkflame drop-in for AMD HIP runtime API
// Exports exactly the symbols expected by ROCm-compiled binaries.

#include "../runtime/device.h"
#include "../runtime/buffer.h"
#include "../runtime/dispatch.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <mutex>

// ── HIP type definitions ──────────────────────────────────────────
typedef int hipError_t;
typedef void *hipStream_t;
typedef void *hipEvent_t;

#define hipSuccess 0
#define hipErrorInvalidValue 1
#define hipErrorMemoryAllocation 2
#define hipErrorInvalidDevicePointer 3
#define hipErrorUnknown 999

// hipMemcpy direction kinds
#define hipMemcpyHostToHost 0
#define hipMemcpyHostToDevice 1
#define hipMemcpyDeviceToHost 2
#define hipMemcpyDeviceToDevice 3
#define hipMemcpyDefault 4

// hipDeviceAttribute_t common values
#define hipDeviceAttributeWarpSize 10
#define hipDeviceAttributeMaxSharedMemoryPerBlock 11
#define hipDeviceAttributeMaxThreadsPerBlock 12
#define hipDeviceAttributeMultiprocessorCount 13
#define hipDeviceAttributeClockRate 14
#define hipDeviceAttributeMemoryClockRate 15
#define hipDeviceAttributeMemoryBusWidth 16
#define hipDeviceAttributeComputeCapabilityMajor 75
#define hipDeviceAttributeComputeCapabilityMinor 76
#define hipDeviceAttributeGcnArch 90

struct hipDeviceProp_t
{
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int gcnArch;
    char gcnArchName[256];
    int isMultiGpuBoard;
    int clockInstructionRate;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    int maxSharedMemoryPerMultiProcessor;
    int isLargeBar;
};

// ── Global ptr→VKFBuffer* map (separate from vkflame_buf_from_ptr) ─
// vkflame_buf_from_ptr uses VkDeviceAddress; here we keep the void* → VKFBuffer* mapping
static std::unordered_map<void *, VKFBuffer *> g_ptr_map;
static std::mutex g_ptr_mutex;

// ── Kernel registry: hostFn* → kernel device name (from __hipRegisterFunction) ──
static std::unordered_map<const void *, std::string> g_kernel_map;
static std::mutex g_kernel_mutex;

static hipError_t g_last_error = hipSuccess;

static void set_last_error(hipError_t e) { g_last_error = e; }

// ── Stream structs ─────────────────────────────────────────────────
struct HIPStream
{
    int pad;
};

// ── dim3 / uint3 — used by HIP kernel-launch ABI stubs ────────────
struct dim3
{
    unsigned x, y, z;
    dim3(unsigned _x = 1, unsigned _y = 1, unsigned _z = 1) : x(_x), y(_y), z(_z) {}
};
struct uint3
{
    unsigned x, y, z;
};

// ── HIP event ──────────────────────────────────────────────────────
struct HIPEvent
{
    bool recorded;
};

extern "C"
{

    // ── Memory ─────────────────────────────────────────────────────────

    hipError_t hipMalloc(void **devPtr, size_t size)
    {
        if (!devPtr)
            return (set_last_error(hipErrorInvalidValue), hipErrorInvalidValue);
        if (size == 0)
        {
            *devPtr = nullptr;
            return hipSuccess;
        }

        VKFBuffer *buf = vkflame_alloc(size);
        if (!buf)
            return (set_last_error(hipErrorMemoryAllocation), hipErrorMemoryAllocation);

        void *raw = reinterpret_cast<void *>(buf->address);
        *devPtr = raw;

        {
            std::lock_guard<std::mutex> lk(g_ptr_mutex);
            g_ptr_map[raw] = buf;
        }

        set_last_error(hipSuccess);
        return hipSuccess;
    }

    hipError_t hipFree(void *devPtr)
    {
        if (!devPtr)
            return hipSuccess;

        VKFBuffer *buf = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_ptr_mutex);
            auto it = g_ptr_map.find(devPtr);
            if (it == g_ptr_map.end())
                return (set_last_error(hipErrorInvalidDevicePointer), hipErrorInvalidDevicePointer);
            buf = it->second;
            g_ptr_map.erase(it);
        }

        vkflame_free(buf);
        set_last_error(hipSuccess);
        return hipSuccess;
    }

    hipError_t hipMallocManaged(void **devPtr, size_t size, unsigned int /*flags*/)
    {
        // Managed memory not separately supported — just allocate GPU memory
        return hipMalloc(devPtr, size);
    }

    hipError_t hipMemcpy(void *dst, const void *src, size_t count, int kind)
    {
        switch (kind)
        {
        case hipMemcpyHostToDevice:
        {
            VKFBuffer *dbuf = nullptr;
            {
                std::lock_guard<std::mutex> lk(g_ptr_mutex);
                auto it = g_ptr_map.find(dst);
                if (it == g_ptr_map.end())
                    return (set_last_error(hipErrorInvalidDevicePointer), hipErrorInvalidDevicePointer);
                dbuf = it->second;
            }
            int rc = vkflame_memcpy_h2d(dbuf, src, count, 0);
            return rc == 0 ? hipSuccess : hipErrorUnknown;
        }
        case hipMemcpyDeviceToHost:
        {
            VKFBuffer *sbuf = nullptr;
            {
                std::lock_guard<std::mutex> lk(g_ptr_mutex);
                auto it = g_ptr_map.find(const_cast<void *>(src));
                if (it == g_ptr_map.end())
                    return (set_last_error(hipErrorInvalidDevicePointer), hipErrorInvalidDevicePointer);
                sbuf = it->second;
            }
            int rc = vkflame_memcpy_d2h(dst, sbuf, count, 0);
            return rc == 0 ? hipSuccess : hipErrorUnknown;
        }
        case hipMemcpyDeviceToDevice:
        {
            VKFBuffer *sbuf = nullptr, *dbuf = nullptr;
            {
                std::lock_guard<std::mutex> lk(g_ptr_mutex);
                auto si = g_ptr_map.find(const_cast<void *>(src));
                auto di = g_ptr_map.find(dst);
                if (si == g_ptr_map.end() || di == g_ptr_map.end())
                    return (set_last_error(hipErrorInvalidDevicePointer), hipErrorInvalidDevicePointer);
                sbuf = si->second;
                dbuf = di->second;
            }
            int rc = vkflame_memcpy_d2d(dbuf, sbuf, count);
            return rc == 0 ? hipSuccess : hipErrorUnknown;
        }
        case hipMemcpyHostToHost:
            memcpy(dst, src, count);
            return hipSuccess;
        default:
            return hipErrorInvalidValue;
        }
    }

    hipError_t hipMemcpyAsync(void *dst, const void *src, size_t count, int kind, hipStream_t /*stream*/)
    {
        // Synchronous implementation (no stream support yet)
        return hipMemcpy(dst, src, count, kind);
    }

    hipError_t hipMemset(void *devPtr, int value, size_t count)
    {
        VKFBuffer *buf = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_ptr_mutex);
            auto it = g_ptr_map.find(devPtr);
            if (it == g_ptr_map.end())
                return (set_last_error(hipErrorInvalidDevicePointer), hipErrorInvalidDevicePointer);
            buf = it->second;
        }
        int rc = vkflame_memset(buf, value, count);
        return rc == 0 ? hipSuccess : hipErrorUnknown;
    }

    hipError_t hipMemsetAsync(void *devPtr, int value, size_t count, hipStream_t /*stream*/)
    {
        return hipMemset(devPtr, value, count);
    }

    // ── Synchronisation ────────────────────────────────────────────────

    hipError_t hipDeviceSynchronize()
    {
        // All our dispatches are already synchronous (submit+wait fence)
        return hipSuccess;
    }

    hipError_t hipStreamCreate(hipStream_t *stream)
    {
        if (!stream)
            return hipErrorInvalidValue;
        *stream = new HIPStream{};
        return hipSuccess;
    }

    hipError_t hipStreamDestroy(hipStream_t stream)
    {
        delete reinterpret_cast<HIPStream *>(stream);
        return hipSuccess;
    }

    hipError_t hipStreamSynchronize(hipStream_t /*stream*/)
    {
        return hipDeviceSynchronize();
    }

    // ── Device queries ─────────────────────────────────────────────────

    hipError_t hipSetDevice(int /*device*/)
    {
        // Single-device runtime
        return hipSuccess;
    }

    hipError_t hipGetDevice(int *device)
    {
        if (device)
            *device = 0;
        return hipSuccess;
    }

    hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int /*device*/)
    {
        if (!prop)
            return hipErrorInvalidValue;
        memset(prop, 0, sizeof(*prop));

        VKFContext *ctx = vkflame_get_context();
        const VKFFeatures &f = ctx->features;

        strncpy(prop->name, f.device_name, 255);
        prop->totalGlobalMem = 16ULL * 1024 * 1024 * 1024; // 16 GB
        prop->warpSize = (int)f.subgroup_size;
        prop->sharedMemPerBlock = (int)f.max_compute_shared_memory;
        prop->maxThreadsPerBlock = 1024;
        prop->multiProcessorCount = 32; // approximate
        prop->major = 12;               // RDNA4
        prop->minor = 1;

        // Report as gfx1201 if AMD
        if (f.vendor_id == 0x1002)
            strncpy(prop->gcnArchName, "gfx1201", 255);
        else
            strncpy(prop->gcnArchName, "unknown", 255);

        prop->gcnArch = 1201;
        return hipSuccess;
    }

    hipError_t hipDeviceGetAttribute(int *value, int attr, int /*device*/)
    {
        if (!value)
            return hipErrorInvalidValue;
        VKFContext *ctx = vkflame_get_context();
        const VKFFeatures &f = ctx->features;

        switch (attr)
        {
        case hipDeviceAttributeWarpSize:
            *value = (int)f.subgroup_size;
            break;
        case hipDeviceAttributeMaxSharedMemoryPerBlock:
            *value = (int)f.max_compute_shared_memory;
            break;
        case hipDeviceAttributeMaxThreadsPerBlock:
            *value = 1024;
            break;
        case hipDeviceAttributeMultiprocessorCount:
            *value = 32;
            break;
        case hipDeviceAttributeComputeCapabilityMajor:
            *value = 12; // RDNA4
            break;
        case hipDeviceAttributeComputeCapabilityMinor:
            *value = 1;
            break;
        case hipDeviceAttributeGcnArch:
            *value = 1201;
            break;
        default:
            *value = 0;
        }
        return hipSuccess;
    }

    // ── Error handling ─────────────────────────────────────────────────

    const char *hipGetErrorString(hipError_t error)
    {
        switch (error)
        {
        case hipSuccess:
            return "hipSuccess";
        case hipErrorInvalidValue:
            return "hipErrorInvalidValue";
        case hipErrorMemoryAllocation:
            return "hipErrorMemoryAllocation";
        case hipErrorInvalidDevicePointer:
            return "hipErrorInvalidDevicePointer";
        default:
            return "hipErrorUnknown";
        }
    }

    hipError_t hipGetLastError() { return g_last_error; }
    hipError_t hipPeekAtLastError() { return g_last_error; }

    // ── Device count ───────────────────────────────────────────────────
    hipError_t hipGetDeviceCount(int *count)
    {
        if (count)
            *count = 1;
        return hipSuccess;
    }

    // ── Driver version ─────────────────────────────────────────────────
    hipError_t hipDriverGetVersion(int *driverVersion)
    {
        if (driverVersion)
            *driverVersion = 60200000; // ROCm 6.2
        return hipSuccess;
    }

    // ── hipGetDevicePropertiesR0600 — ROCm 6 ABI variant ────────────────
    // The R0600 struct is larger than the old one; we write fields at their
    // known byte offsets to avoid needing the exact struct definition.
    hipError_t hipGetDevicePropertiesR0600(void *prop, int /*device*/)
    {
        if (!prop)
            return hipErrorInvalidValue;
        memset(prop, 0, 2048); // ROCm 6 hipDeviceProp_t is ~1450 bytes
        VKFContext *ctx = vkflame_get_context();
        const VKFFeatures &f = ctx->features;
        uint8_t *p = reinterpret_cast<uint8_t *>(prop);
        // name[256]                 at   0
        strncpy(reinterpret_cast<char *>(p + 0), f.device_name, 255);
        // (uuid[16], luid[8], luidMask[4], pad[4]) then totalGlobalMem at 288
        *reinterpret_cast<size_t *>(p + 288) = 16ULL * 1024 * 1024 * 1024;  // 16 GB
        *reinterpret_cast<size_t *>(p + 296) = f.max_compute_shared_memory; // sharedMemPerBlock
        *reinterpret_cast<int *>(p + 304) = 65536;                          // regsPerBlock
        *reinterpret_cast<int *>(p + 308) = (int)f.subgroup_size;           // warpSize ← critical
        *reinterpret_cast<int *>(p + 320) = 1024;                           // maxThreadsPerBlock
        *reinterpret_cast<int *>(p + 360) = 12;                             // major = 12 (RDNA4)
        *reinterpret_cast<int *>(p + 364) = 1;                              // minor = 1
        *reinterpret_cast<int *>(p + 388) = 32;                             // multiProcessorCount
        // gcnArchName[256] at offset 1160: after reserved[63] + hipReserved[32]
        if (f.vendor_id == 0x1002)
            strncpy(reinterpret_cast<char *>(p + 1160), "gfx1201", 255);
        return hipSuccess;
    }

    // ── Device reset ───────────────────────────────────────────────────
    hipError_t hipDeviceReset() { return hipSuccess; }

    // ── Peer access stubs ──────────────────────────────────────────────
    hipError_t hipDeviceCanAccessPeer(int *canAccessPeer, int /*dev*/, int /*peer*/)
    {
        if (canAccessPeer)
            *canAccessPeer = 0;
        return hipSuccess;
    }
    hipError_t hipDeviceEnablePeerAccess(int /*peerDevice*/, unsigned int /*flags*/) { return hipSuccess; }
    hipError_t hipDeviceDisablePeerAccess(int /*peerDevice*/) { return hipSuccess; }

    // ── Events ─────────────────────────────────────────────────────────
    hipError_t hipEventCreateWithFlags(hipEvent_t *event, unsigned int /*flags*/)
    {
        if (!event)
            return hipErrorInvalidValue;
        *event = new HIPEvent{false};
        return hipSuccess;
    }
    hipError_t hipEventDestroy(hipEvent_t event)
    {
        delete reinterpret_cast<HIPEvent *>(event);
        return hipSuccess;
    }
    hipError_t hipEventRecord(hipEvent_t event, hipStream_t /*stream*/)
    {
        if (event)
            reinterpret_cast<HIPEvent *>(event)->recorded = true;
        return hipSuccess;
    }
    hipError_t hipEventSynchronize(hipEvent_t /*event*/) { return hipDeviceSynchronize(); }
    hipError_t hipEventElapsedTime(float *ms, hipEvent_t /*start*/, hipEvent_t /*stop*/)
    {
        if (ms)
            *ms = 0.0f;
        return hipSuccess;
    }

    // ── Stream with flags ──────────────────────────────────────────────
    hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int /*flags*/)
    {
        return hipStreamCreate(stream);
    }
    hipError_t hipStreamWaitEvent(hipStream_t /*stream*/, hipEvent_t /*event*/, unsigned int /*flags*/)
    {
        return hipSuccess;
    }

    // ── Stream capture stubs ───────────────────────────────────────────
    hipError_t hipStreamBeginCapture(hipStream_t /*stream*/, int /*mode*/) { return hipSuccess; }
    hipError_t hipStreamEndCapture(hipStream_t /*stream*/, void **graph)
    {
        if (graph)
            *graph = nullptr;
        return hipSuccess;
    }
    hipError_t hipGraphDestroy(void * /*graph*/) { return hipSuccess; }

    // ── Host-pinned memory — backed by heap, staged through Vulkan ────
    hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int /*flags*/)
    {
        if (!ptr)
            return hipErrorInvalidValue;
        *ptr = malloc(size);
        return *ptr ? hipSuccess : hipErrorMemoryAllocation;
    }
    hipError_t hipHostFree(void *ptr)
    {
        free(ptr);
        return hipSuccess;
    }
    hipError_t hipHostRegister(void * /*ptr*/, size_t /*size*/, unsigned int /*flags*/) { return hipSuccess; }
    hipError_t hipHostUnregister(void * /*ptr*/) { return hipSuccess; }

    // ── Memory info ────────────────────────────────────────────────────
    hipError_t hipMemGetInfo(size_t *free_out, size_t *total_out)
    {
        const size_t gb16 = 16ULL * 1024 * 1024 * 1024;
        if (free_out)
            *free_out = gb16;
        if (total_out)
            *total_out = gb16;
        return hipSuccess;
    }

    // ── Memory advise (NUMA hint — noop) ──────────────────────────────
    hipError_t hipMemAdvise(const void * /*ptr*/, size_t /*count*/, int /*advice*/, int /*dev*/)
    {
        return hipSuccess;
    }

    // ── 2D async copy (row-by-row fallback) ───────────────────────────
    hipError_t hipMemcpy2DAsync(void *dst, size_t dpitch,
                                const void *src, size_t spitch,
                                size_t width, size_t height,
                                int kind, hipStream_t /*stream*/)
    {
        for (size_t row = 0; row < height; row++)
            hipMemcpy(reinterpret_cast<char *>(dst) + row * dpitch,
                      reinterpret_cast<const char *>(src) + row * spitch,
                      width, kind);
        return hipSuccess;
    }

    // ── Peer async copy ────────────────────────────────────────────────
    hipError_t hipMemcpyPeerAsync(void *dst, int /*dstDev*/,
                                  const void *src, int /*srcDev*/,
                                  size_t count, hipStream_t stream)
    {
        return hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToDevice, stream);
    }

    // ── hipLaunchKernel — full dispatch table ─────────────────────────
    // Decodes args using ggml-hip kernel calling conventions (from ggml source analysis).
    // args[i] is a void* pointing to the i-th argument value.
    hipError_t hipLaunchKernel(const void *func,
                               dim3 gridDim, dim3 blockDim,
                               void **args, size_t /*sharedMem*/,
                               hipStream_t /*stream*/)
    {
        if (!func || !args)
            return hipSuccess;

        // Look up registered kernel name
        std::string kname;
        {
            std::lock_guard<std::mutex> lk(g_kernel_mutex);
            auto it = g_kernel_map.find(func);
            if (it != g_kernel_map.end())
                kname = it->second;
        }
        if (kname.empty())
            return hipSuccess; // unrecognised — silently skip

        // ── Arg helpers ─────────────────────────────────────────────────
        // Device pointer arg: args[i] points to a void* (raw GPU address)
        auto raw_ptr = [&](int i) -> void *
        {
            return *reinterpret_cast<void **>(args[i]);
        };
        // Integer arg
        auto get_i32 = [&](int i) -> int32_t
        {
            return *reinterpret_cast<int32_t *>(args[i]);
        };
        auto get_i64 = [&](int i) -> int64_t
        {
            return *reinterpret_cast<int64_t *>(args[i]);
        };
        auto get_f32 = [&](int i) -> float
        {
            return *reinterpret_cast<float *>(args[i]);
        };
        // VKFBuffer from a device-pointer arg
        auto get_buf = [&](int i) -> VKFBuffer *
        {
            void *p = raw_ptr(i);
            std::lock_guard<std::mutex> lk(g_ptr_mutex);
            auto it = g_ptr_map.find(p);
            return it != g_ptr_map.end() ? it->second : nullptr;
        };

        auto has = [&](const char *s)
        {
            return kname.find(s) != std::string::npos;
        };

        VKFContext *ctx = vkflame_get_context();

        // ── Dequantisation kernels ────────────────────────────────────────
        // q4_0, q4_1, q5_0: (vx, y, int nb32)  grid.x ≈ (nb32+7)/8
        // q8_0:              (vx, y, int nb)     grid.x = nb
        // q4_K, q5_K, q6_K: (vx, yy)            nb = gridDim.x
        if (has("q4_0") || has("q4K_0"))
        {
            VKFBuffer *src = get_buf(0), *dst = get_buf(1);
            if (src && dst)
                vkflame_dispatch_dequant(ctx, src, dst,
                                         (uint32_t)get_i32(2), VKF_DEQUANT_Q4_0);
        }
        else if (has("q4_1"))
        {
            VKFBuffer *src = get_buf(0), *dst = get_buf(1);
            if (src && dst)
                vkflame_dispatch_dequant(ctx, src, dst,
                                         (uint32_t)get_i32(2), VKF_DEQUANT_Q4_1);
        }
        else if (has("q5_0") || has("q5K_0"))
        {
            VKFBuffer *src = get_buf(0), *dst = get_buf(1);
            if (src && dst)
                vkflame_dispatch_dequant(ctx, src, dst,
                                         (uint32_t)get_i32(2), VKF_DEQUANT_Q5_0);
        }
        else if (has("q8_0"))
        {
            VKFBuffer *src = get_buf(0), *dst = get_buf(1);
            if (src && dst)
            {
                // nb can be int or int64 depending on template instantiation
                uint32_t nb = (blockDim.x == 32) ? (uint32_t)get_i32(2)
                                                 : (uint32_t)(get_i64(2) / 32);
                vkflame_dispatch_dequant(ctx, src, dst, nb, VKF_DEQUANT_Q8_0);
            }
        }
        else if (has("q4_K") || has("q4K"))
        {
            VKFBuffer *src = get_buf(0), *dst = get_buf(1);
            if (src && dst)
                vkflame_dispatch_dequant(ctx, src, dst,
                                         gridDim.x, VKF_DEQUANT_Q4_K);
        }
        else if (has("q5_K") || has("q5K"))
        {
            VKFBuffer *src = get_buf(0), *dst = get_buf(1);
            if (src && dst)
                vkflame_dispatch_dequant(ctx, src, dst,
                                         gridDim.x, VKF_DEQUANT_Q5_K);
        }
        else if (has("q6_K") || has("q6K"))
        {
            VKFBuffer *src = get_buf(0), *dst = get_buf(1);
            if (src && dst)
                vkflame_dispatch_dequant(ctx, src, dst,
                                         gridDim.x, VKF_DEQUANT_Q6_K);
        }
        // ── Element-wise unary (in-place supported: src may == dst) ─────
        // ggml pattern: (const float* x, float* dst, int k)
        else if (has("silu_f32") || has("silu"))
        {
            void *src = raw_ptr(0), *dst = raw_ptr(1);
            uint32_t k = (uint32_t)get_i32(2);
            vkflame_dispatch_elementwise_f32(ctx, src, dst, k, VKF_EW_SILU);
        }
        else if (has("gelu_quick") || has("gelu_erf") || has("gelu"))
        {
            void *src = raw_ptr(0), *dst = raw_ptr(1);
            uint32_t k = (uint32_t)get_i32(2);
            uint32_t op = has("quick") ? VKF_EW_GELU_QUICK : VKF_EW_GELU_ERF;
            vkflame_dispatch_elementwise_f32(ctx, src, dst, k, op);
        }
        else if (has("relu"))
        {
            void *src = raw_ptr(0), *dst = raw_ptr(1);
            uint32_t k = (uint32_t)get_i32(2);
            vkflame_dispatch_elementwise_f32(ctx, src, dst, k, VKF_EW_RELU);
        }
        else if (has("tanh_f32") || (has("tanh") && !has("atanh")))
        {
            void *src = raw_ptr(0), *dst = raw_ptr(1);
            uint32_t k = (uint32_t)get_i32(2);
            vkflame_dispatch_elementwise_f32(ctx, src, dst, k, VKF_EW_TANH);
        }
        else if (has("sigmoid"))
        {
            void *src = raw_ptr(0), *dst = raw_ptr(1);
            uint32_t k = (uint32_t)get_i32(2);
            vkflame_dispatch_elementwise_f32(ctx, src, dst, k, VKF_EW_SIGMOID);
        }
        else if (has("sqr_f32") || has("sqr"))
        {
            void *src = raw_ptr(0), *dst = raw_ptr(1);
            uint32_t k = (uint32_t)get_i32(2);
            vkflame_dispatch_elementwise_f32(ctx, src, dst, k, VKF_EW_SQR);
        }
        else if (has("sqrt_f32") || has("sqrt"))
        {
            void *src = raw_ptr(0), *dst = raw_ptr(1);
            uint32_t k = (uint32_t)get_i32(2);
            vkflame_dispatch_elementwise_f32(ctx, src, dst, k, VKF_EW_SQRT);
        }
        else if (has("neg_f32") || has("neg"))
        {
            void *src = raw_ptr(0), *dst = raw_ptr(1);
            uint32_t k = (uint32_t)get_i32(2);
            vkflame_dispatch_elementwise_f32(ctx, src, dst, k, VKF_EW_NEG);
        }
        else if (has("abs_f32") || has("abs"))
        {
            void *src = raw_ptr(0), *dst = raw_ptr(1);
            uint32_t k = (uint32_t)get_i32(2);
            vkflame_dispatch_elementwise_f32(ctx, src, dst, k, VKF_EW_ABS);
        }
        // ── Scale ─────────────────────────────────────────────────────────
        // ggml scale_f32 kernel:
        //   (const float* x, float* dst, float scale, float bias, int64_t nelements)
        else if (has("scale_f32") || has("scale_f16"))
        {
            void *src = raw_ptr(0), *dst = raw_ptr(1);
            float sc = get_f32(2);
            float bias = get_f32(3);
            uint32_t n = (uint32_t)get_i64(4);
            vkflame_dispatch_scale_f32(ctx, src, dst, n, sc, bias);
        }
        // ── RMS norm fp32 ─────────────────────────────────────────────────
        // ggml rms_norm_f32 kernel:
        //   (const float* x, float* dst, int ncols, int s_row, int s_chan, int s_sample, float eps)
        else if (has("rms_norm_f32") || has("rms_norm"))
        {
            void *src = raw_ptr(0), *dst = raw_ptr(1);
            int32_t ncols = get_i32(2);
            float eps = get_f32(6);
            uint32_t nrows = gridDim.x;
            vkflame_dispatch_rms_norm_f32(ctx, src, dst, nrows, (uint32_t)ncols, eps);
        }
        // ── RoPE neox ────────────────────────────────────────────────────
        // ggml rope_neox kernel: (x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale,
        //                         p_delta_rows, ext_factor, attn_factor,
        //                         corr_dims_0, corr_dims_1, theta_scale, ...)
        else if (has("rope") && (has("neox") || has("llama") || has("norm")))
        {
            void *src = raw_ptr(0);
            void *dst_ptr = raw_ptr(1);
            int32_t ne0 = get_i32(2);
            int32_t ne1 = get_i32(3);
            int32_t n_dims = get_i32(6);
            void *pos = raw_ptr(7);
            float freq_scale = get_f32(8);
            float theta_scale = get_f32(13);
            uint32_t n_seqs = gridDim.y > 0 ? gridDim.y : 1;
            vkflame_dispatch_rope_neox(ctx, src, pos, dst_ptr,
                                       (uint32_t)ne0, (uint32_t)ne1, (uint32_t)n_dims, n_seqs,
                                       theta_scale, freq_scale, 0, 0);
        }
        // ── Binary ops — add, mul, sub, div ─────────────────────────────
        // ggml k_bin_bcast kernel: (src0, src1, dst, ...)
        else if (has("bin_bcast") || has("add_f32") || has("mul_f32") ||
                 has("sub_f32") || has("div_f32"))
        {
            void *src0 = raw_ptr(0), *src1 = raw_ptr(1), *dst = raw_ptr(2);
            uint32_t n = gridDim.x * blockDim.x;
            uint32_t op = VKF_BINOP_ADD;
            if (has("mul"))
                op = VKF_BINOP_MUL;
            else if (has("sub"))
                op = VKF_BINOP_SUB;
            else if (has("div"))
                op = VKF_BINOP_DIV;
            vkflame_dispatch_binop_f32(ctx, src0, src1, dst, n, n, op);
        }
        else
        {
            fprintf(stderr, "[vkflame] hipLaunchKernel: unhandled kernel: %s\n",
                    kname.c_str());
        }

        return hipSuccess;
    }

    // ── HIP fat-binary registration ───────────────────────────────────
    void **__hipRegisterFatBinary(const void * /*data*/)
    {
        static int dummy = 0;
        return reinterpret_cast<void **>(&dummy);
    }
    void __hipUnregisterFatBinary(void ** /*modules*/) {}

    // Store hostFn → deviceName mapping so hipLaunchKernel can dispatch correctly
    void __hipRegisterFunction(
        void ** /*modules*/,
        const void *hostFn,
        char * /*deviceFn*/,
        const char *deviceName,
        int /*threadLimit*/,
        uint3 * /*tid*/, uint3 * /*bid*/,
        dim3 * /*blockDim*/, dim3 * /*gridDim*/, int * /*wSize*/)
    {
        if (hostFn && deviceName)
        {
            std::lock_guard<std::mutex> lk(g_kernel_mutex);
            g_kernel_map[hostFn] = std::string(deviceName);
        }
    }

    hipError_t __hipPushCallConfiguration(dim3 /*gridDim*/, dim3 /*blockDim*/,
                                          size_t /*sharedMem*/, hipStream_t /*stream*/)
    {
        return hipSuccess;
    }
    hipError_t __hipPopCallConfiguration(dim3 * /*gridDim*/, dim3 * /*blockDim*/,
                                         size_t * /*sharedMem*/, hipStream_t * /*stream*/)
    {
        return hipSuccess;
    }

    // ── Occupancy query stub ───────────────────────────────────────────
    hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(
        int *numBlocks, const void * /*func*/, int blockSize, size_t /*dynSharedMemPerBlk*/)
    {
        // Report a reasonable value — ggml uses this for launch configuration
        if (numBlocks)
            *numBlocks = 65536 / (blockSize > 0 ? blockSize : 256);
        return hipSuccess;
    }

} // extern "C"
