#version 460

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly  buffer XBuf      { float x_data[]; };
layout(set = 0, binding = 1) writeonly buffer ValBuf    { float out_vals[]; };
layout(set = 0, binding = 2) writeonly buffer IdxBuf    { uint  out_idx[]; };

layout(push_constant) uniform PC {
    uint M;       // rows
    uint N;       // columns
    uint K;       // number of top-k values to find
    uint largest; // 1 = largest (top-k), 0 = smallest (bottom-k)
} pc;

shared float s_val[256];
shared uint  s_idx[256];

void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;

    if (row >= pc.M) return;

    uint row_base = row * pc.N;

    // Load row into shared memory (using -inf for out-of-bounds)
    // If N > 256, we need to handle overflow — each thread loads one element
    // For N <= 256, direct load. For N > 256, load first 256 (limitation: K <= 256)
    if (tid < pc.N) {
        float val = x_data[row_base + tid];
        // For smallest-k: negate to reuse top-k logic
        s_val[tid] = (pc.largest == 0u) ? -val : val;
        s_idx[tid] = tid;
    } else {
        s_val[tid] = -1e38; // sentinel for unused slots
        s_idx[tid] = tid;
    }
    barrier();
    memoryBarrierShared();

    // Selection sort — K iterations (finds top-K in order)
    for (uint k = 0u; k < pc.K; k++) {
        // Find max in range [k, N-1] using parallel reduction
        // Use barrier-based tournament
        uint search_len = pc.N - k;
        if (tid == 0u && search_len > 0u) {
            // Thread 0 finds the max
            uint max_i = k;
            float max_v = s_val[k];
            for (uint i = k + 1u; i < pc.N && i < 256u; i++) {
                if (s_val[i] > max_v) {
                    max_v = s_val[i];
                    max_i = i;
                }
            }
            // Swap max into position k
            float tmp_v = s_val[k];
            uint  tmp_i = s_idx[k];
            s_val[k]     = s_val[max_i];
            s_idx[k]     = s_idx[max_i];
            s_val[max_i] = tmp_v;
            s_idx[max_i] = tmp_i;
        }
        barrier();
        memoryBarrierShared();
    }

    // Write K results
    if (tid < pc.K) {
        // Undo negation for smallest-k
        float final_val = (pc.largest == 0u) ? -s_val[tid] : s_val[tid];
        out_vals[row * pc.K + tid] = final_val;
        out_idx [row * pc.K + tid] = s_idx[tid];
    }
}
