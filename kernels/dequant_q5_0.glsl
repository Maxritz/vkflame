// dequant_q5_0.glsl — Q5_0 → float32
// Block format: [fp16 d (2B)][qh[4] (4B)][qs[16] (16B)] = 22 bytes, 32 weights per block
// 5 bits per weight: lower 4 from qs nibbles, high bit from qh
// actual = d * (5bit_value - 16)
// Launch: gridDim.x = (nb32+7)/8, blockDim.x = 32
#version 460
layout(local_size_x = 32) in;

layout(set = 0, binding = 0) readonly  buffer InBuf  { uint data[]; } src;
layout(set = 0, binding = 1) writeonly buffer OutBuf { float data[]; } dst;

layout(push_constant) uniform PC { uint nb32; } pc;

uint byte_at(uint b) { return (src.data[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu; }
uint u16_at (uint b) { return byte_at(b) | (byte_at(b + 1u) << 8u); }
uint u32_at (uint b) { return byte_at(b) | (byte_at(b+1u)<<8u) | (byte_at(b+2u)<<16u) | (byte_at(b+3u)<<24u); }

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint il  = tid / 8u;   // 0..3
    uint ir  = tid % 8u;   // 0..7
    uint i   = gl_WorkGroupID.x;
    uint ib  = 8u * i + ir;
    if (ib >= pc.nb32) return;

    // Q5_0 block: [fp16 d][qh[4]][qs[16]] = 22 bytes
    uint base = ib * 22u;
    float d   = unpackHalf2x16(u16_at(base)).x;
    float dm  = -16.0 * d;

    uint qh = u32_at(base + 2u);   // 32 high bits packed

    uint q_base   = base + 6u + 4u * il;
    uint out_base = 256u * i + 32u * ir + 4u * il;

    for (uint l = 0u; l < 4u; l++) {
        uint q    = byte_at(q_base + l);
        uint iqs  = 4u * il + l;   // index into qs nibble array
        // Q5_0 high bit extraction (from dequantize_q5_0 in dequantize.cuh)
        uint xh0 = ((qh >> (iqs + 0u)) << 4u) & 0x10u;
        uint xh1 = ((qh >> (iqs + 12u))      ) & 0x10u;
        float x0 = float((q & 0xFu) | xh0);
        float x1 = float((q >> 4u)  | xh1);
        dst.data[out_base + l]        = x0 * d + dm;
        dst.data[out_base + l + 16u]  = x1 * d + dm;
    }
}
