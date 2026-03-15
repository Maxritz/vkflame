#version 460
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_16bit_storage       : require

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly  buffer X     { float16_t x[];   };
layout(set = 0, binding = 1) readonly  buffer Gamma { float16_t g[];   };
layout(set = 0, binding = 2) writeonly buffer Y     { float16_t y[];   };

layout(push_constant) uniform PC {
    uint  M;
    uint  N;
    float eps;
} pc;

// One slot per subgroup: max 256/8 = 32 subgroups when subgroupSize=8 (worst case)
shared float s_sg_sums[32];
shared float s_rrms;

void main() {
    uint row   = gl_WorkGroupID.x;
    uint tid   = gl_LocalInvocationID.x;
    uint sg_id = gl_SubgroupID;

    if (row >= pc.M) return;

    // Pass 1: each thread accumulates sum of squares over its slice of columns
    float sum_sq = 0.0;
    for (uint c = tid; c < pc.N; c += 256u) {
        float val = float(x[row * pc.N + c]);
        sum_sq += val * val;
    }

    // Subgroup reduce — each subgroup sums its lanes
    float sg_sum = subgroupAdd(sum_sq);

    // Subgroup leader stores partial sum into shared memory
    if (subgroupElect()) {
        s_sg_sums[sg_id] = sg_sum;
    }
    barrier();
    memoryBarrierShared();

    // Thread 0 accumulates all subgroup partial sums and computes rrms
    if (tid == 0) {
        float total = 0.0;
        uint n_sg = gl_NumSubgroups;
        for (uint s = 0u; s < n_sg; s++) {
            total += s_sg_sums[s];
        }
        // rrms = 1 / sqrt(mean_square + eps)
        s_rrms = inversesqrt(total / float(pc.N) + pc.eps);
    }
    barrier();
    memoryBarrierShared();

    // Pass 2: normalise and scale by gamma
    float rrms = s_rrms;
    for (uint c = tid; c < pc.N; c += 256u) {
        float val   = float(x[row * pc.N + c]);
        float gamma = float(g[c]);
        y[row * pc.N + c] = float16_t(val * rrms * gamma);
    }
}
