#version 460
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_shuffle    : require

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly  buffer X { float x[]; };
layout(set = 0, binding = 1) writeonly buffer Y { float y[]; };

layout(push_constant) uniform PC {
    uint M;
    uint N;
} pc;

// One slot per subgroup for (max, d) pairs
shared float s_sg_max[32];
shared float s_sg_d[32];
shared float s_final_max;
shared float s_final_d;

void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint sg_id = gl_SubgroupID;

    if (row >= pc.M) return;
    uint base = row * pc.N;

    // Pass 1 — online max+d accumulation per thread
    float m = -1e38;
    float d = 0.0;
    for (uint c = tid; c < pc.N; c += 256u) {
        float xi = x[base + c];
        float m_new = max(m, xi);
        // rescale previous d by exp(m - m_new), add new term
        d = d * exp(m - m_new) + exp(xi - m_new);
        m = m_new;
    }

    // Subgroup reduction: merge (m, d) pairs across lanes
    // Use iterative approach: reduce via subgroupShuffleXor
    for (uint offset = gl_SubgroupSize / 2u; offset > 0u; offset >>= 1u) {
        float m_b = subgroupShuffleXor(m, offset);
        float d_b = subgroupShuffleXor(d, offset);
        float m_new = max(m, m_b);
        float d_new = d * exp(m - m_new) + d_b * exp(m_b - m_new);
        m = m_new;
        d = d_new;
    }

    // Subgroup leader stores results
    if (subgroupElect()) {
        s_sg_max[sg_id] = m;
        s_sg_d[sg_id]   = d;
    }
    barrier();
    memoryBarrierShared();

    // Thread 0 merges all subgroups
    if (tid == 0) {
        float gm = s_sg_max[0];
        float gd = s_sg_d[0];
        uint n_sg = gl_NumSubgroups;
        for (uint s = 1u; s < n_sg; s++) {
            float m_b = s_sg_max[s];
            float d_b = s_sg_d[s];
            float m_new = max(gm, m_b);
            float d_new = gd * exp(gm - m_new) + d_b * exp(m_b - m_new);
            gm = m_new;
            gd = d_new;
        }
        s_final_max = gm;
        s_final_d   = gd;
    }
    barrier();
    memoryBarrierShared();

    // Pass 2 — write softmax output: exp(xi - max) / d
    float final_max = s_final_max;
    float final_d   = s_final_d;
    for (uint c = tid; c < pc.N; c += 256u) {
        y[base + c] = exp(x[base + c] - final_max) / final_d;
    }
}
