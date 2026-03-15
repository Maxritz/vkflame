// dequant_q4_k.glsl — Q4_K → float32
// Block: [fp16 d (2B)][fp16 dmin (2B)][scales[12] (12B)][qs[128] (128B)] = 144 bytes, 256 weights
// Launch: gridDim.x = nb (one workgroup per block), blockDim.x = 32
#version 460
layout(local_size_x = 32) in;

layout(set = 0, binding = 0) readonly  buffer InBuf  { uint data[]; } src;
layout(set = 0, binding = 1) writeonly buffer OutBuf { float data[]; } dst;

layout(push_constant) uniform PC { uint nb; } pc;

uint byte_at(uint b) { return (src.data[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu; }
uint u16_at (uint b) { return byte_at(b) | (byte_at(b + 1u) << 8u); }

// Extract 6-bit scale and min from packed scales array (mirrors get_scale_min_k4 in ggml)
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
    uint tid = gl_LocalInvocationID.x;   // 0..31
    uint il  = tid / 8u;                 // 0..3
    uint ir  = tid % 8u;                 // 0..7
    uint is  = 2u * il;                  // scale pair index
    uint ib  = gl_WorkGroupID.x;
    if (ib >= pc.nb) return;

    // Q4_K layout: [fp16 d][fp16 dmin][scales[12]][qs[128]] = 144 bytes
    uint base        = ib * 144u;
    float dall       = unpackHalf2x16(u16_at(base)).x;
    float dmin_val   = unpackHalf2x16(u16_at(base + 2u)).x;
    uint  scales_off = base + 4u;
    uint  qs_off     = base + 16u;     // 4 + 12 = 16

    float sc0, m0, sc1, m1;
    get_scale_min_k4(is + 0u, scales_off, sc0, m0);
    get_scale_min_k4(is + 1u, scales_off, sc1, m1);

    float d0   = dall * sc0;   float min0 = dmin_val * m0;
    float d1   = dall * sc1;   float min1 = dmin_val * m1;

    // qs pointer: qs + 32*il + 4*ir = qs_off + 32*il + 4*ir
    uint q_off    = qs_off + 32u * il + 4u * ir;
    // output: ib*256 + 64*il + 4*ir
    uint out_base = ib * 256u + 64u * il + 4u * ir;

    for (uint l = 0u; l < 4u; l++) {
        uint q = byte_at(q_off + l);
        dst.data[out_base + l]        = d0 * float(q & 0xFu) - min0;
        dst.data[out_base + l + 32u]  = d1 * float(q >> 4u)  - min1;
    }
}
