// dequant_q5_k.glsl — Q5_K → float32
// Block: [fp16 d (2B)][fp16 dmin (2B)][scales[12] (12B)][qh[32] (32B)][qs[128] (128B)] = 176 bytes, 256 weights
// 5 bits per weight: lower 4 from qs nibbles, high bit from qh
// Launch: gridDim.x = nb (one workgroup per block), blockDim.x = 64
#version 460
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly  buffer InBuf  { uint data[]; } src;
layout(set = 0, binding = 1) writeonly buffer OutBuf { float data[]; } dst;

layout(push_constant) uniform PC { uint nb; } pc;

uint byte_at(uint b) { return (src.data[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu; }
uint u16_at (uint b) { return byte_at(b) | (byte_at(b + 1u) << 8u); }

void get_scale_min_k4(uint j, uint sbase, out float sc, out float mn) {
    if (j < 4u) {
        sc = float(byte_at(sbase + j) & 63u);
        mn = float(byte_at(sbase + j + 4u) & 63u);
    } else {
        sc = float((byte_at(sbase + j + 4u) & 0xFu) | ((byte_at(sbase + j - 4u) >> 6u) << 4u));
        mn = float((byte_at(sbase + j + 4u) >> 4u)  | ((byte_at(sbase + j) >> 6u) << 4u));
    }
}

void main() {
    uint tid = gl_LocalInvocationID.x;   // 0..63
    uint il  = tid / 16u;                // 0..3
    uint ir  = tid % 16u;                // 0..15
    uint is  = 2u * il;
    uint ib  = gl_WorkGroupID.x;
    if (ib >= pc.nb) return;

    // Q5_K layout (176 bytes):
    // [fp16 d (2)][fp16 dmin (2)][scales[12] (12)][qh[32] (32)][qs[128] (128)]
    // offsets:   0                2                4             16              48
    uint base       = ib * 176u;
    float dall      = unpackHalf2x16(u16_at(base)).x;
    float dmin_val  = unpackHalf2x16(u16_at(base + 2u)).x;
    uint scales_off = base + 4u;
    uint qh_off     = base + 16u;
    uint qs_off     = base + 48u;

    float sc0, m0, sc1, m1;
    get_scale_min_k4(is + 0u, scales_off, sc0, m0);
    get_scale_min_k4(is + 1u, scales_off, sc1, m1);

    float d0   = dall * sc0;   float min0 = dmin_val * m0;
    float d1   = dall * sc1;   float min1 = dmin_val * m1;

    // ql = qs + 32*il + 2*ir;   qh = qh + 2*ir
    uint ql_off = qs_off + 32u * il + 2u * ir;
    uint qh_ptr = qh_off + 2u * ir;

    uint out_base = ib * 256u + 64u * il + 2u * ir;

    // hm = 1 << (2*il)
    uint hm  = 1u << (2u * il);

    uint ql0 = byte_at(ql_off + 0u);
    uint ql1 = byte_at(ql_off + 1u);
    uint qh0 = byte_at(qh_ptr + 0u);
    uint qh1 = byte_at(qh_ptr + 1u);

    float y0 = d0 * float((ql0 & 0xFu) + ((qh0 & hm) != 0u ? 16u : 0u)) - min0;
    float y1 = d0 * float((ql1 & 0xFu) + ((qh1 & hm) != 0u ? 16u : 0u)) - min0;
    hm = hm << 1u;
    float y32 = d1 * float((ql0 >> 4u) + ((qh0 & hm) != 0u ? 16u : 0u)) - min1;
    float y33 = d1 * float((ql1 >> 4u) + ((qh1 & hm) != 0u ? 16u : 0u)) - min1;

    dst.data[out_base + 0u]  = y0;
    dst.data[out_base + 1u]  = y1;
    dst.data[out_base + 32u] = y32;
    dst.data[out_base + 33u] = y33;
}
