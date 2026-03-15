#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require

// Specialisation constants — NOT push constants
layout(constant_id = 0) const int HEAD_DIM  = 64;
layout(constant_id = 1) const int TILE_SIZE = 16;

layout(local_size_x = 64) in;

// Q [B * Hq * Sq * D], K [B * Hkv * Skv * D], V [B * Hkv * Skv * D]
layout(set = 0, binding = 0) readonly buffer Q_buf { float16_t Q[]; };
layout(set = 0, binding = 1) readonly buffer K_buf { float16_t K[]; };
layout(set = 0, binding = 2) readonly buffer V_buf { float16_t V[]; };
layout(set = 0, binding = 3) writeonly buffer O_buf { float16_t O[]; };

layout(push_constant) uniform PC {
    uint  B;
    uint  Hq;
    uint  Hkv;
    uint  Sq;
    uint  Skv;
    uint  D;       // head dim (== HEAD_DIM specialisation constant)
    float scale;   // 1/sqrt(D)
    uint  is_causal;
} pc;

// LDS — must use HEAD_DIM (specialisation constant), not a literal
shared float s_o[HEAD_DIM];  // accumulator for output vector
shared float s_m;            // running max
shared float s_d;            // running normalisation factor
shared float s_kv[TILE_SIZE * HEAD_DIM]; // K or V tile loaded into LDS

void main() {
    // Each workgroup handles one query position
    // gl_WorkGroupID: x = q_pos, y = q_head, z = batch
    uint q_pos  = gl_WorkGroupID.x;
    uint q_head = gl_WorkGroupID.y;
    uint batch  = gl_WorkGroupID.z;
    uint tid    = gl_LocalInvocationID.x;

    if (q_pos >= pc.Sq || q_head >= pc.Hq || batch >= pc.B) return;

    // GQA: map query head to key/value head
    int kv_head = int(q_head) / (int(pc.Hq) / int(pc.Hkv));

    uint q_base = ((batch * pc.Hq + q_head) * pc.Sq + q_pos) * pc.D;
    uint kv_base_k = ((batch * pc.Hkv + uint(kv_head)) * pc.Skv) * pc.D;
    uint kv_base_v = kv_base_k;
    uint o_base  = q_base;

    // Initialise accumulators in shared memory
    if (tid < uint(HEAD_DIM)) {
        s_o[tid] = 0.0;
    }
    if (tid == 0) {
        s_m = -1e38;
        s_d = 0.0;
    }
    barrier();
    memoryBarrierShared();

    // Iterate over KV tiles
    uint n_tiles = (pc.Skv + uint(TILE_SIZE) - 1u) / uint(TILE_SIZE);

    for (uint tile = 0u; tile < n_tiles; tile++) {
        uint kv_tile_start = tile * uint(TILE_SIZE);

        // Causal masking: skip tile if all positions are after q_pos
        if (pc.is_causal != 0u && kv_tile_start > q_pos) {
            continue;  // skip entire tile
        }

        // ── Compute QK^T scores for this tile ──────────────────
        // Each thread handles one QK dot product (one kv position in tile)
        float score = -1e38;
        uint kv_pos_in_tile = tid % uint(TILE_SIZE);
        uint kv_pos = kv_tile_start + kv_pos_in_tile;
        bool valid_pos = (kv_pos < pc.Skv);

        // Causal: mask out positions after q_pos within tile
        if (pc.is_causal != 0u && kv_pos > q_pos) valid_pos = false;

        if (valid_pos && tid < uint(TILE_SIZE)) {
            float dot_val = 0.0;
            uint k_offset = (kv_base_k + kv_pos * pc.D);
            for (uint d = 0u; d < pc.D; d++) {
                dot_val += float(Q[q_base + d]) * float(K[k_offset + d]);
            }
            score = dot_val * pc.scale;
        }

        // Online softmax update — find max score in tile
        // Use simple sequential approach: thread 0 computes the online update
        // after loading scores into LDS (simplified, not fully parallel)
        if (tid < uint(TILE_SIZE)) {
            s_kv[tid] = valid_pos ? score : -1e38;
        }
        barrier();
        memoryBarrierShared();

        if (tid == 0) {
            float m_old = s_m;
            float d_old = s_d;

            // Find max score in this tile (for numerical stability)
            float tile_max = -1e38;
            uint tile_count = min(uint(TILE_SIZE), pc.Skv - kv_tile_start);
            for (uint t = 0u; t < tile_count; t++) {
                uint kv_t = kv_tile_start + t;
                if (pc.is_causal != 0u && kv_t > q_pos) continue;
                tile_max = max(tile_max, s_kv[t]);
            }

            // Online softmax update formula:
            //   m_new = max(m_old, tile_max)
            //   d_new = d_old * exp(m_old - m_new) + sum_t(exp(score_t - m_new))
            float m_new = max(m_old, tile_max);
            float rescale = exp(m_old - m_new);  // factor to rescale old accum

            // Rescale accumulated output by exp(m_old - m_new) — done once per tile
            for (uint d = 0u; d < uint(HEAD_DIM); d++) {
                s_o[d] *= rescale;
            }

            float d_new = d_old * rescale;

            // Accumulate weighted V for each position in this tile
            for (uint t = 0u; t < tile_count; t++) {
                uint kv_t = kv_tile_start + t;
                if (pc.is_causal != 0u && kv_t > q_pos) continue;
                float exp_score = exp(s_kv[t] - m_new);
                d_new += exp_score;

                uint v_offset = kv_base_v + kv_t * pc.D;
                for (uint d = 0u; d < uint(HEAD_DIM); d++) {
                    s_o[d] += exp_score * float(V[v_offset + d]);
                }
            }

            s_m = m_new;
            s_d = d_new;
        }
        barrier();
        memoryBarrierShared();
    }

    // Write output: s_o[j] / s_d
    if (tid < uint(HEAD_DIM)) {
        O[o_base + tid] = float16_t(s_o[tid] / s_d);
    }
}
