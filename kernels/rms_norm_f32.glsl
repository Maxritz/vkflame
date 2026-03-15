// rms_norm_f32.glsl — RMS layer normalization on float32 (no gamma, ggml variant)
// y = x / sqrt(mean(x^2) + eps)
// Launch: gridDim.x = M (one workgroup per row), blockDim.x = 256
#version 460
#extension GL_KHR_shader_subgroup_arithmetic : require
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly  buffer SrcBuf { float data[]; } src;
layout(set = 0, binding = 1) writeonly buffer DstBuf { float data[]; } dst;

layout(push_constant) uniform PC {
    uint  M;
    uint  N;
    float eps;
} pc;

// Enough slots for up to 16 subgroups (256 threads / 16 threads-per-subgroup minimum)
shared float s_sg[16];
shared float s_rrms;

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.M) return;

    uint tid      = gl_LocalInvocationID.x;
    uint row_base = row * pc.N;

    // 1. Each thread computes partial sum of squared elements
    float sum_sq = 0.0;
    for (uint col = tid; col < pc.N; col += 256u) {
        float v = src.data[row_base + col];
        sum_sq += v * v;
    }

    // 2. Subgroup reduce → subgroup leader stores in shared memory
    float sg_sum = subgroupAdd(sum_sq);
    if (subgroupElect()) {
        s_sg[gl_SubgroupID] = sg_sum;
    }

    barrier();
    memoryBarrierShared();

    // 3. Thread 0 reduces subgroup sums and computes rrms
    if (tid == 0u) {
        float total = 0.0;
        for (uint s = 0u; s < gl_NumSubgroups; s++) total += s_sg[s];
        s_rrms = inversesqrt(total / float(pc.N) + pc.eps);
    }

    barrier();
    memoryBarrierShared();

    // 4. Apply normalization
    float rrms = s_rrms;
    for (uint col = tid; col < pc.N; col += 256u) {
        uint idx = row_base + col;
        dst.data[idx] = src.data[idx] * rrms;
    }
}
