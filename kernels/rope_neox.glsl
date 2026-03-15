// rope_neox.glsl — Rotary Position Embedding (neox / llama format) on float32
// Pairs element j with element j + ne0/2 (neox layout)
// theta_j = pos * theta_scale^j  where theta_scale = pow(freq_base, -2/n_dims)
// Launch: gridDim.x = ne1*n_seqs, gridDim.y = (ne0/2+63)/64, blockDim.x = 64
#version 460
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly  buffer SrcBuf { float data[]; } src;
layout(set = 0, binding = 1) readonly  buffer PosBuf { int  data[]; } pos_buf;
layout(set = 0, binding = 2) writeonly buffer DstBuf { float data[]; } dst;

layout(push_constant) uniform PC {
    uint  ne0;          // head dimension (total)
    uint  ne1;          // number of heads per sequence
    uint  n_dims;       // number of rotary dimensions (<= ne0)
    uint  n_seqs;       // number of sequences in batch
    float theta_scale;  // pow(freq_base, -2.0/n_dims) — rotation speed per dimension
    float freq_scale;   // global frequency multiplier (default 1.0)
    uint  pad0;
    uint  pad1;
} pc;

void main() {
    // Global layout: gl_WorkGroupID.x = row (head_idx in [0..ne1*n_seqs))
    //                gl_WorkGroupID.y = dim_pair group
    uint dim_pair = gl_WorkGroupID.y * 64u + gl_LocalInvocationID.x;  // j in [0..ne0/2)
    uint row      = gl_WorkGroupID.x;   // absolute row index across sequences

    if (dim_pair >= pc.ne0 / 2u) return;
    if (row >= pc.ne1 * pc.n_seqs) return;

    uint row_in_seq  = row % pc.ne1;         // head index within sequence
    uint seq_idx     = row / pc.ne1;

    // Source/dest with contiguous layout: element [seq][head][dim]
    uint ix_base = seq_idx * pc.ne1 * pc.ne0 + row_in_seq * pc.ne0;
    uint x0_idx  = ix_base + dim_pair;
    uint x1_idx  = ix_base + dim_pair + pc.ne0 / 2u;

    uint d0_idx  = row * pc.ne0 + dim_pair;
    uint d1_idx  = row * pc.ne0 + dim_pair + pc.ne0 / 2u;

    // Passthrough dimensions beyond n_dims
    if (dim_pair * 2u >= pc.n_dims) {
        dst.data[d0_idx] = src.data[x0_idx];
        dst.data[d1_idx] = src.data[x1_idx];
        return;
    }

    // theta = position * theta_scale^dim_pair * freq_scale
    int pos_val = pos_buf.data[seq_idx];
    float theta = float(pos_val) * pow(pc.theta_scale, float(dim_pair)) * pc.freq_scale;

    float cos_t = cos(theta);
    float sin_t = sin(theta);

    float x0 = src.data[x0_idx];
    float x1 = src.data[x1_idx];

    dst.data[d0_idx] = x0 * cos_t - x1 * sin_t;
    dst.data[d1_idx] = x0 * sin_t + x1 * cos_t;
}
