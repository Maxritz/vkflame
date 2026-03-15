// hip_runtime_shim.cpp — vkflame drop-in for AMD HIP runtime API
// Exports exactly the symbols expected by ROCm-compiled binaries.

#include "../runtime/device.h"
#include "../runtime/buffer.h"
#include "../runtime/dispatch.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <string>
#include <unordered_map>
#include <mutex>
#ifdef _WIN32
#include <windows.h>
#endif

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

// ── GFX arch name resolution ─────────────────────────────────────────
// Priority:
//  1. VKFLAME_GFX_ARCH env var (explicit override, e.g. "gfx1100")
//  2. HSA_OVERRIDE_GFX_VERSION env var  (e.g. "11.0.0" -> "gfx1100")
//  3. "gfx1201" fallback (RDNA4 default)
static std::string vkf_resolve_gfx_arch()
{
    // 1. Explicit vkflame override
    const char *vkf = getenv("VKFLAME_GFX_ARCH");
    if (vkf && vkf[0])
        return std::string(vkf);

    // 2. HSA_OVERRIDE_GFX_VERSION: "major.minor.stepping" -> "gfxMmS"
    const char *hsa = getenv("HSA_OVERRIDE_GFX_VERSION");
    if (hsa && hsa[0])
    {
        int maj = 0, min = 0, step = 0;
        if (sscanf(hsa, "%d.%d.%d", &maj, &min, &step) >= 2)
        {
            char buf[32];
            snprintf(buf, sizeof(buf), "gfx%d%d%d", maj, min, step);
            return std::string(buf);
        }
    }

    // 3. Fallback
    return "gfx1201";
}

// Parse major/minor from a gfx arch string (e.g. "gfx1201" -> major=12, minor=1)
static void vkf_parse_gfx_major_minor(const std::string &arch, int &major, int &minor)
{
    major = 12;
    minor = 1; // safe defaults
    // arch format: "gfx" + digits, e.g. gfx1201 = maj 12, min 0, step 1
    // We treat the first 1-2 digits as major (stopping when remaining <= 2 digits)
    const char *p = arch.c_str();
    if (p[0] == 'g' && p[1] == 'f' && p[2] == 'x')
        p += 3;
    int val = atoi(p);
    if (val >= 1000)
    {
        major = val / 100;
        minor = (val % 100) / 10;
    }
    else if (val >= 100)
    {
        major = val / 100;
        minor = val % 100;
    }
    else
    {
        major = val;
        minor = 0;
    }
}

// ── Global ptr→VKFBuffer* map (separate from vkflame_buf_from_ptr) ─
// vkflame_buf_from_ptr uses VkDeviceAddress; here we keep the void* → VKFBuffer* mapping
static std::unordered_map<void *, VKFBuffer *> g_ptr_map;
static std::mutex g_ptr_mutex;

// ── Debug logging — raw Win32 API, no CRT dependency, works from DllMain ────────
// Always writes to C:\vkflame_loaded.log so we can confirm DLL load regardless
// of environment variables. Verbose per-call log goes to C:\vkflame_debug.log.
static HANDLE g_debug_file = INVALID_HANDLE_VALUE;
static FILE *g_log_file = nullptr;
static std::mutex g_log_mutex;

static void vkf_log_init()
{
    static bool inited = false;
    if (inited)
        return;
    inited = true;

    // Write a "loaded" marker using raw Win32 — no CRT, works in DllMain.
    // GetTempPathA is safe from DllMain and always writable (no admin needed).
    char tmpdir[MAX_PATH];
    GetTempPathA(MAX_PATH, tmpdir);
    char proofpath[MAX_PATH];
    wsprintfA(proofpath, "%svkflame_loaded.log", tmpdir);
    HANDLE hProof = CreateFileA(proofpath,
                                GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hProof != INVALID_HANDLE_VALUE)
    {
        char buf[256];
        DWORD written;
        int len = wsprintfA(buf, "vkflame loaded pid=%lu\r\n",
                            (unsigned long)GetCurrentProcessId());
        SetFilePointer(hProof, 0, nullptr, FILE_END);
        WriteFile(hProof, buf, (DWORD)len, &written, nullptr);
        CloseHandle(hProof);
    }

    // Also open a CRT log for verbose per-call output (debug mode).
    const char *tmp = tmpdir; // already obtained via GetTempPathA above
    char path[512];
    snprintf(path, sizeof(path), "%svkflame_debug.log", tmp);
    g_log_file = fopen(path, "a");
    if (g_log_file)
    {
        const char *dbg = getenv("VKFLAME_DEBUG");
        fprintf(g_log_file, "\n=== vkflame DLL loaded (pid=%lu) VKFLAME_DEBUG=%s ===\n",
                (unsigned long)GetCurrentProcessId(), dbg ? dbg : "(not set)");
        fflush(g_log_file);
    }
}

static void vkf_log(const char *fmt, ...)
{
    std::lock_guard<std::mutex> lk(g_log_mutex);
    // Only write to stderr if it looks like a valid console/pipe handle.
    // Writing to an invalid handle in a subprocess causes Go's runtime to emit
    // "failed to get console mode for stderr: The handle is invalid." which is
    // confusing. Check with a zero-byte write first.
    HANDLE hstderr = GetStdHandle(STD_ERROR_HANDLE);
    if (hstderr != INVALID_HANDLE_VALUE && hstderr != nullptr)
    {
        DWORD dummy;
        if (WriteFile(hstderr, "", 0, &dummy, nullptr))
        {
            va_list ap1;
            va_start(ap1, fmt);
            vfprintf(stderr, fmt, ap1);
            va_end(ap1);
        }
    }
    if (!g_log_file)
        return;
    va_list ap2;
    va_start(ap2, fmt);
    vfprintf(g_log_file, fmt, ap2);
    va_end(ap2);
    fflush(g_log_file);
}

// ── Kernel registry: hostFn* → kernel device name (from __hipRegisterFunction) ──
static std::unordered_map<const void *, std::string> g_kernel_map;
static std::mutex g_kernel_mutex;

// ── Lazy Vulkan init — safe to call from any HIP function ─────────────────────
// The discovery runner never calls vkflame_init() explicitly, so we must do it
// lazily. If Vulkan init fails we still return plausible device info so Ollama
// picks the ROCm backend instead of falling back to CPU.
static bool g_vkf_inited = false;
static int g_vkf_init_result = -1;
static std::mutex g_vkf_init_mutex;

static int vkf_ensure_init()
{
    std::lock_guard<std::mutex> lk(g_vkf_init_mutex);
    if (g_vkf_inited)
        return g_vkf_init_result;
    g_vkf_inited = true;
    vkf_log_init(); // ensure log file is open before we write to it
    vkf_log("[vkflame] vkflame_init() starting (pid=%lu)...\n",
            (unsigned long)GetCurrentProcessId());
    g_vkf_init_result = vkflame_init();
    vkf_log("[vkflame] vkflame_init() returned %d\n", g_vkf_init_result);
    return g_vkf_init_result;
}

// Returns the context, or nullptr if Vulkan could not be initialised.
static VKFContext *vkf_ctx()
{
    vkf_ensure_init();
    return vkflame_get_context();
}

// Fallback device name / arch when Vulkan init failed.
// Reads HSA_OVERRIDE_GFX_VERSION to give Ollama something useful.
static std::string vkf_fallback_arch()
{
    const char *ov = getenv("HSA_OVERRIDE_GFX_VERSION");
    if (ov && ov[0])
    {
        // Convert "12.0.1" → "gfx1201"
        int ma = 0, mi = 0, pa = 0;
        if (sscanf(ov, "%d.%d.%d", &ma, &mi, &pa) >= 2)
        {
            char buf[32];
            snprintf(buf, sizeof(buf), "gfx%d%d%d", ma, mi, pa);
            return buf;
        }
    }
    return "gfx1201"; // RDNA4 default
}

// hipModuleLoad / hipModuleGetFunction / hipModuleLaunchKernel path.
// Used by AOTriton, pre-compiled HSACO code objects, and HIP driver API callers.
typedef void *hipModule_t;
typedef void *hipFunction_t;

struct HipModuleImpl
{
    std::string name; // source path or "<data>"
};
struct HipFunctionImpl
{
    std::string kernel_name; // device kernel name
};

// Global registry: hipFunction_t handle → kernel name
static std::unordered_map<hipFunction_t, std::string> g_module_func_map;
static std::mutex g_module_func_mutex;

static hipFunction_t make_func_handle(const char *name)
{
    auto *impl = new HipFunctionImpl{std::string(name)};
    {
        std::lock_guard<std::mutex> lk(g_module_func_mutex);
        g_module_func_map[impl] = impl->kernel_name;
    }
    // Also register in g_kernel_map so hipLaunchKernel can dispatch it
    {
        std::lock_guard<std::mutex> lk(g_kernel_mutex);
        g_kernel_map[impl] = impl->kernel_name;
    }
    return impl;
}

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
        vkf_log_init();
        vkf_log("[vkflame] hipMalloc(size=%zu)\n", size);
        vkf_ensure_init();
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

        vkf_log("[vkflame]   -> hipMalloc ptr=%p rc=%s\n", buf ? reinterpret_cast<void *>(buf->address) : nullptr, buf ? "ok" : "FAIL");
        set_last_error(hipSuccess);
        return hipSuccess;
    }

    hipError_t hipFree(void *devPtr)
    {
        vkf_log_init();
        vkf_log("[vkflame] hipFree(ptr=%p)\n", devPtr);
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

    hipError_t hipSetDevice(int device)
    {
        vkf_log_init();
        vkf_log("[vkflame] hipSetDevice(%d)\n", device);
        return hipSuccess;
    }

    hipError_t hipGetDevice(int *device)
    {
        vkf_log_init();
        vkf_log("[vkflame] hipGetDevice()\n");
        if (device)
            *device = 0;
        return hipSuccess;
    }

    hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int device)
    {
        vkf_log_init();
        vkf_log("[vkflame] hipGetDeviceProperties(dev=%d)\n", device);
        if (!prop)
            return hipErrorInvalidValue;
        memset(prop, 0, sizeof(*prop));

        VKFContext *ctx = vkf_ctx();
        std::string arch;
        int warpSize = 32;
        uint32_t sharedMem = 65536;

        if (ctx)
        {
            const VKFFeatures &f = ctx->features;
            strncpy(prop->name, f.device_name, 255);
            warpSize = (int)f.subgroup_size;
            sharedMem = f.max_compute_shared_memory;
            arch = (f.vendor_id == 0x1002) ? vkf_resolve_gfx_arch() : "unknown";
        }
        else
        {
            arch = vkf_fallback_arch();
            strncpy(prop->name, "AMD Radeon GPU (vkflame)", 255);
        }

        prop->totalGlobalMem = 16ULL * 1024 * 1024 * 1024;
        prop->warpSize = warpSize;
        prop->sharedMemPerBlock = sharedMem;
        prop->maxThreadsPerBlock = 1024;
        prop->multiProcessorCount = 32;
        strncpy(prop->gcnArchName, arch.c_str(), 255);
        int maj = 12, min = 1;
        vkf_parse_gfx_major_minor(arch, maj, min);
        prop->major = maj;
        prop->minor = min;
        prop->gcnArch = atoi(arch.c_str() + (arch.size() > 3 ? 3 : 0));
        vkf_log("[vkflame]   -> name=\"%s\" gcnArch=%s major=%d minor=%d warpSize=%d\n",
                prop->name, arch.c_str(), prop->major, prop->minor, prop->warpSize);
        return hipSuccess;
    }

    hipError_t hipDeviceGetAttribute(int *value, int attr, int device)
    {
        vkf_log_init();
        vkf_log("[vkflame] hipDeviceGetAttribute(attr=%d, dev=%d)\n", attr, device);
        if (!value)
            return hipErrorInvalidValue;
        VKFContext *ctx = vkf_ctx();
        // Safe defaults used when Vulkan not yet initialised
        uint32_t subgroup = ctx ? ctx->features.subgroup_size : 32;
        uint32_t sharedMem = ctx ? ctx->features.max_compute_shared_memory : 65536;
        uint32_t vendorId = ctx ? ctx->features.vendor_id : 0x1002;
        (void)vendorId;
        struct VKFFeatures _fb{};
        if (!ctx)
        {
            _fb.subgroup_size = subgroup;
            _fb.max_compute_shared_memory = sharedMem;
        }
        const VKFFeatures &f = ctx ? ctx->features : _fb;

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
        {
            int maj = 12, min = 1;
            vkf_parse_gfx_major_minor(vkf_resolve_gfx_arch(), maj, min);
            *value = maj;
            break;
        }
        case hipDeviceAttributeComputeCapabilityMinor:
        {
            int maj = 12, min = 1;
            vkf_parse_gfx_major_minor(vkf_resolve_gfx_arch(), maj, min);
            *value = min;
            break;
        }
        case hipDeviceAttributeGcnArch:
        {
            std::string arch = vkf_resolve_gfx_arch();
            *value = atoi(arch.c_str() + (arch.size() > 3 ? 3 : 0));
            break;
        }
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
        vkf_log_init();
        vkf_log("[vkflame] hipGetDeviceCount()\n");
        if (count)
            *count = 1;
        return hipSuccess;
    }

    // ── Driver version ─────────────────────────────────────────────────
    hipError_t hipDriverGetVersion(int *driverVersion)
    {
        vkf_log_init();
        vkf_log("[vkflame] hipDriverGetVersion()\n");
        if (driverVersion)
            *driverVersion = 60200000; // ROCm 6.2
        return hipSuccess;
    }

    // ── hipGetDevicePropertiesR0600 — ROCm 6 ABI variant ────────────────
    // The R0600 struct is larger than the old one; we write fields at their
    // known byte offsets to avoid needing the exact struct definition.
    hipError_t hipGetDevicePropertiesR0600(void *prop, int device)
    {
        vkf_log_init();
        vkf_log("[vkflame] hipGetDevicePropertiesR0600(dev=%d)\n", device);
        if (!prop)
            return hipErrorInvalidValue;
        memset(prop, 0, 2048); // ROCm 6 hipDeviceProp_t is ~1450 bytes
        VKFContext *ctx = vkf_ctx();
        uint8_t *p = reinterpret_cast<uint8_t *>(prop);

        std::string arch;
        int warpSize = 32;
        size_t sharedMem = 65536;
        const char *devname = "AMD Radeon GPU (vkflame)";

        if (ctx)
        {
            const VKFFeatures &f = ctx->features;
            devname = f.device_name;
            warpSize = (int)f.subgroup_size;
            sharedMem = f.max_compute_shared_memory;
            arch = (f.vendor_id == 0x1002) ? vkf_resolve_gfx_arch() : "unknown";
        }
        else
        {
            arch = vkf_fallback_arch();
        }

        // name[256] at 0
        strncpy(reinterpret_cast<char *>(p + 0), devname, 255);
        // totalGlobalMem at 288
        *reinterpret_cast<size_t *>(p + 288) = 16ULL * 1024 * 1024 * 1024;
        *reinterpret_cast<size_t *>(p + 296) = sharedMem; // sharedMemPerBlock
        *reinterpret_cast<int *>(p + 304) = 65536;        // regsPerBlock
        *reinterpret_cast<int *>(p + 308) = warpSize;     // warpSize ← critical
        *reinterpret_cast<int *>(p + 320) = 1024;         // maxThreadsPerBlock
        {
            int maj = 12, min = 1;
            vkf_parse_gfx_major_minor(arch, maj, min);
            *reinterpret_cast<int *>(p + 360) = maj; // major
            *reinterpret_cast<int *>(p + 364) = min; // minor
            *reinterpret_cast<int *>(p + 388) = 32;  // multiProcessorCount
            // gcnArchName[256] at offset 1160
            strncpy(reinterpret_cast<char *>(p + 1160), arch.c_str(), 255);
            vkf_log("[vkflame]   -> R0600 name=\"%s\" gcnArch=%s major=%d minor=%d warpSize=%d\n",
                    devname, arch.c_str(), maj, min, warpSize);
        }
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
        vkf_log_init();
        const size_t gb16 = 16ULL * 1024 * 1024 * 1024;
        if (free_out)
            *free_out = gb16;
        if (total_out)
            *total_out = gb16;
        vkf_log("[vkflame] hipMemGetInfo() -> free=%zu total=%zu\n", gb16, gb16);
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

        if (g_log_file)
            vkf_log("[vkflame] kernel: %s  grid=(%u,%u,%u) block=(%u,%u,%u)\n",
                    kname.c_str(), gridDim.x, gridDim.y, gridDim.z,
                    blockDim.x, blockDim.y, blockDim.z);
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

        VKFContext *ctx = vkf_ctx();
        if (!ctx)
        {
            // Vulkan init failed — cannot dispatch GPU kernels.
            // Return success so the caller doesn't abort; work is silently skipped.
            return hipSuccess;
        }

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
        // ── Softmax ────────────────────────────────────────────────────────
        // ggml: soft_max_f32(x, mask, dst, ncols_x, nrows_x, nrows_y, scale, max_bias, m0, m1, n_head_log2)
        // or:   soft_max_f32(x, dst, ncols, ...)
        else if (has("soft_max") || has("softmax"))
        {
            void *src = raw_ptr(0);
            // ggml ≥ 0.10: soft_max_f32(x, mask, dst, ncols, nrows, ...)
            // ggml <  0.10: soft_max_f32(x, dst, ncols, nrows, ...)
            // Detect by inspecting whether arg[1] looks like a mask pointer or a dst.
            // We dispatch grid.y rows x grid.x cols so nrows=gridDim.y, ncols=gridDim.x*blockDim.x
            uint32_t ncols = get_i32(2);
            uint32_t nrows = gridDim.y > 0 ? gridDim.y : 1;
            void *dst;
            if (ncols > 0)
            {
                // New signature: soft_max_f32(x, mask, dst, ncols, ...)
                dst = raw_ptr(2);
                ncols = (uint32_t)get_i32(3);
            }
            else
            {
                // Old signature: soft_max_f32(x, dst, ncols, nrows, ...)
                dst = raw_ptr(1);
                ncols = (uint32_t)get_i32(2);
            }
            if (ncols < 1)
                ncols = blockDim.x; // fallback
            vkflame_dispatch_softmax(ctx, src, dst, (int)nrows, (int)ncols);
        }
        // ── mul_mat (quantized GEMM — the #1 LLM hot path) ────────────────
        // ggml-hip launches mul_mat_q*_q8_1 kernels for quantized mat-mul.
        // args: (vx /*weights quant*/, vy /*activations q8_1*/, dst /*float32*/, ncols_x, nrows_x,
        //        ncols_y, nrows_y, ncols_dst, channel, channel_x)
        // We dequantise the weight block into a temp buffer, dequantise the Q8_1 activations,
        // then run FP32 GEMM into the output buffer.
        else if (has("mul_mat") && (has("q4_0") || has("q4_1") || has("q4_K") ||
                                    has("q5_0") || has("q5_K") || has("q6_K") ||
                                    has("q8_0") || has("iq2") || has("iq3") || has("iq4")))
        {
            // IQ2/IQ3/IQ4 are not yet implemented — skip gracefully
            if (has("iq2") || has("iq3") || has("iq4"))
            {
                vkf_log("[vkflame] mul_mat: IQ2/IQ3/IQ4 not yet implemented (%s), skipping\n",
                        kname.c_str());
                return hipSuccess;
            }

            // vx = weight (quantized), vy = activation (Q8_1), dst = output FP32
            VKFBuffer *w_quant = get_buf(0);
            VKFBuffer *act = get_buf(1);
            VKFBuffer *dst_buf = get_buf(2);
            if (!w_quant || !act || !dst_buf)
            {
                vkf_log("[vkflame] mul_mat: unresolved buffers for %s\n", kname.c_str());
                return hipSuccess;
            }
            int32_t ncols_x = get_i32(3); // K (weight columns = activation rows)
            int32_t nrows_x = get_i32(4); // M (weight rows = output rows)
            int32_t nrows_y = get_i32(6); // N (batch / sequence tokens)
            if (ncols_x < 1 || nrows_x < 1 || nrows_y < 1)
                return hipSuccess;

            // Step 1: dequantise weight matrix to FP32
            uint32_t n_elements = (uint32_t)(nrows_x * ncols_x);
            VKFBuffer *w_f32 = vkflame_alloc((size_t)n_elements * sizeof(float));
            int quant_type = VKF_DEQUANT_Q4_0;
            if (has("q4_1"))
                quant_type = VKF_DEQUANT_Q4_1;
            else if (has("q8_0"))
                quant_type = VKF_DEQUANT_Q8_0;
            else if (has("q5_0"))
                quant_type = VKF_DEQUANT_Q5_0;
            else if (has("q4_K"))
                quant_type = VKF_DEQUANT_Q4_K;
            else if (has("q5_K"))
                quant_type = VKF_DEQUANT_Q5_K;
            else if (has("q6_K"))
                quant_type = VKF_DEQUANT_Q6_K;

            // block size depends on quant type
            uint32_t block_size = (quant_type == VKF_DEQUANT_Q4_K || quant_type == VKF_DEQUANT_Q5_K) ? 256u
                                  : (quant_type == VKF_DEQUANT_Q6_K)                                 ? 256u
                                                                                                     : 32u;
            uint32_t n_blocks = (n_elements + block_size - 1) / block_size;
            vkflame_dispatch_dequant(ctx, w_quant, w_f32, n_blocks, quant_type);

            // Step 1b: dequantise Q8_1 activations -> FP32
            // Q8_1 format: 36 bytes/block (fp16 d + fp16 s + 32×int8_t qs)
            uint32_t act_elements = (uint32_t)(nrows_y * ncols_x);
            VKFBuffer *act_f32 = vkflame_alloc((size_t)act_elements * sizeof(float));
            uint32_t act_blocks = (act_elements + 31u) / 32u;
            vkflame_dispatch_dequant(ctx, act, act_f32, act_blocks, VKF_DEQUANT_Q8_1);

            // Step 2: GEMM — output[nrows_y, nrows_x] = act_f32[nrows_y, ncols_x] @ w_f32^T[ncols_x, nrows_x]
            vkflame_dispatch_linear(ctx,
                                    (void *)act_f32->address, (void *)w_f32->address,
                                    nullptr, (void *)dst_buf->address,
                                    nrows_y, nrows_x, ncols_x,
                                    VKF_DTYPE_FP16_FP32OUT, 0, 1 /*transB*/, 0,
                                    nullptr, nullptr, nullptr);

            vkflame_free(act_f32);
            vkflame_free(w_f32);
            vkf_log("[vkflame] mul_mat %s  M=%d N=%d K=%d\n",
                    kname.c_str(), nrows_y, nrows_x, ncols_x);
        }
        // ── mul_mat fp16 × fp16 ────────────────────────────────────────────
        else if (has("mul_mat") && (has("f16") || has("fp16")))
        {
            VKFBuffer *A = get_buf(0), *B = get_buf(1), *D = get_buf(2);
            if (!A || !B || !D)
                return hipSuccess;
            int32_t M = get_i32(6); // nrows_y (batch tokens)
            int32_t N = get_i32(4); // nrows_x (output features)
            int32_t K = get_i32(3); // ncols_x (input features)
            if (M < 1 || N < 1 || K < 1)
                return hipSuccess;
            vkflame_dispatch_linear(ctx, (void *)A->address, (void *)B->address,
                                    nullptr, (void *)D->address,
                                    M, N, K, VKF_DTYPE_FP16, 0, 1, 0,
                                    nullptr, nullptr, nullptr);
        }
        // ── get_rows — embedding lookup ───────────────────────────────────
        // ggml: get_rows_f32(src0, src1, dst, ne00, nb01, nb1, s1, s2)
        //   src0 = weight table [vocab × dim], src1 = indices [batch], dst = output [batch × dim]
        else if (has("get_rows") || has("getrows"))
        {
            VKFBuffer *weight = get_buf(0);
            VKFBuffer *idx = get_buf(1);
            void *dst = raw_ptr(2);
            if (!weight || !idx)
                return hipSuccess;
            int32_t dim = get_i32(3); // ne00 = embedding dim
            int32_t vocab = (int32_t)(weight->size / ((size_t)dim * sizeof(float)));
            int32_t batch = (int32_t)(idx->size / sizeof(int32_t));
            vkflame_dispatch_embedding(ctx,
                                       (void *)weight->address, (void *)idx->address, dst,
                                       vocab, dim, batch);
        }
        else
        {
            vkf_log("[vkflame] hipLaunchKernel: unhandled kernel: %s\n",
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

    // ── hipModule* — HIP driver API (used by AOTriton, pre-compiled HSACO) ──────
    // We don't actually execute .hsaco GPU code — we parse the kernel names and
    // route matching names through our Vulkan dispatch table, same as hipLaunchKernel.

    hipError_t hipModuleLoad(hipModule_t *module, const char *fname)
    {
        if (!module)
            return hipErrorInvalidValue;
        auto *m = new HipModuleImpl{fname ? fname : "<unknown>"};
        *module = m;
        if (getenv("VKFLAME_DEBUG"))
            vkf_log("[vkflame] hipModuleLoad: %s\n", m->name.c_str());
        return hipSuccess;
    }

    hipError_t hipModuleLoadData(hipModule_t *module, const void * /*image*/)
    {
        if (!module)
            return hipErrorInvalidValue;
        auto *m = new HipModuleImpl{"<data>"};
        *module = m;
        return hipSuccess;
    }

    hipError_t hipModuleLoadDataEx(hipModule_t *module, const void * /*image*/,
                                   unsigned int /*numOptions*/, int * /*options*/,
                                   void ** /*optionValues*/)
    {
        return hipModuleLoadData(module, nullptr);
    }

    hipError_t hipModuleUnload(hipModule_t module)
    {
        delete reinterpret_cast<HipModuleImpl *>(module);
        return hipSuccess;
    }

    hipError_t hipModuleGetFunction(hipFunction_t *func, hipModule_t /*mod*/, const char *name)
    {
        if (!func || !name)
            return hipErrorInvalidValue;
        *func = make_func_handle(name);
        if (getenv("VKFLAME_DEBUG"))
            vkf_log("[vkflame] hipModuleGetFunction: %s\n", name);
        return hipSuccess;
    }

    // hipModuleLaunchKernel: dispatch via the same table as hipLaunchKernel
    hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                     uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                                     uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                                     uint32_t /*sharedMem*/, hipStream_t stream,
                                     void **kernelParams, void ** /*extra*/)
    {
        dim3 grid(gridX, gridY, gridZ);
        dim3 block(blockX, blockY, blockZ);
        return hipLaunchKernel(f, grid, block, kernelParams, 0, stream);
    }

    // AMD extension variant — same signature but reversed extra/kernelParams order
    hipError_t hipExtModuleLaunchKernel(hipFunction_t f,
                                        uint32_t globalX, uint32_t globalY, uint32_t globalZ,
                                        uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                                        uint32_t /*sharedMem*/, hipStream_t stream,
                                        void **kernelParams, void ** /*extra*/,
                                        hipEvent_t /*startEvent*/, hipEvent_t /*stopEvent*/,
                                        uint32_t /*flags*/)
    {
        // globalX/Y/Z here are total threads, not blocks — convert to grid dims
        uint32_t gridX = (globalX + blockX - 1) / blockX;
        uint32_t gridY = (globalY + blockY - 1) / blockY;
        uint32_t gridZ = (globalZ + blockZ - 1) / blockZ;
        dim3 grid(gridX ? gridX : 1, gridY ? gridY : 1, gridZ ? gridZ : 1);
        dim3 block(blockX, blockY, blockZ);
        return hipLaunchKernel(f, grid, block, kernelParams, 0, stream);
    }

    // ── Function attribute query stubs ────────────────────────────────
    struct hipFuncAttributes
    {
        size_t sharedSizeBytes;
        size_t constSizeBytes;
        size_t localSizeBytes;
        int maxThreadsPerBlock;
        int numRegs;
        int ptxVersion;
        int binaryVersion;
        int cacheModeCA;
        int maxDynamicSharedSizeBytes;
        int preferredShmemCarveout;
    };

    hipError_t hipFuncGetAttributes(hipFuncAttributes *attr, const void * /*func*/)
    {
        if (!attr)
            return hipErrorInvalidValue;
        memset(attr, 0, sizeof(*attr));
        attr->maxThreadsPerBlock = 1024;
        attr->numRegs = 32;
        return hipSuccess;
    }

    hipError_t hipFuncSetAttribute(const void * /*func*/, int /*attr*/, int /*value*/)
    {
        return hipSuccess;
    }

    hipError_t hipFuncSetCacheConfig(const void * /*func*/, int /*cacheConfig*/)
    {
        return hipSuccess;
    }

    // ── hipRTC stubs (runtime compilation — not supported, return error) ─
    // AOTriton uses pre-compiled HSACO and does NOT call hipRTC, but some
    // Python torch.compile paths may try it. Return "not supported" gracefully.
    hipError_t hiprtcCreateProgram(void ** /*prog*/, const char * /*src*/,
                                   const char * /*name*/, int /*numHeaders*/,
                                   const char ** /*headers*/, const char ** /*includeNames*/)
    {
        return hipErrorUnknown; // hiprtcResult: HIPRTC_ERROR_COMPILATION
    }

    // ── Device pointer info ───────────────────────────────────────────
    hipError_t hipPointerGetAttributes(void *attributes, const void *ptr)
    {
        // attributes is hipPointerAttribute_t — just zero-fill (device memory type)
        if (!attributes)
            return hipErrorInvalidValue;
        // hipPointerAttribute_t is layout-compatible with a small struct;
        // zero = hipMemoryTypeDevice, which is what callers check for.
        memset(attributes, 0, 64);
        // Set memoryType field at offset 0 to hipMemoryTypeDevice (1)
        *reinterpret_cast<int *>(attributes) = 1;
        (void)ptr;
        return hipSuccess;
    }

    hipError_t hipDrvPointerGetAttributes(unsigned int /*numAttributes*/,
                                          int * /*attributes*/,
                                          void ** /*data*/,
                                          const void * /*ptr*/)
    {
        return hipSuccess;
    }

} // extern "C"

// ── DLL entry point — fires on load, before any exported function ──────────
#ifdef _WIN32
BOOL WINAPI DllMain(HINSTANCE /*hinstDLL*/, DWORD fdwReason, LPVOID /*lpvReserved*/)
{
    if (fdwReason == DLL_PROCESS_ATTACH)
    {
        vkf_log_init();
    }
    return TRUE;
}
#endif
