// dequant_q4_1.glsl — Q4_1 → float32
// Block format: [fp16 d (2B)][fp16 m (2B)][qs[16] (16B)] = 20 bytes, 32 weights per block
// actual = d * nibble + m  (no centering offset)
// Launch: gridDim.x = (nb32+7)/8, blockDim.x = 32
#version 460
layout(local_size_x = 32) in;

layout(set = 0, binding = 0) readonly  buffer InBuf  { uint data[]; } src;
layout(set = 0, binding = 1) writeonly buffer OutBuf { float data[]; } dst;

layout(push_constant) uniform PC { uint nb32; } pc;

uint  byte_at(uint b) { return (src.data[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu; }
uint  u16_at (uint b) { return byte_at(b) | (byte_at(b + 1u) << 8u); }

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint il  = tid / 8u;
    uint ir  = tid % 8u;
    uint i   = gl_WorkGroupID.x;
    uint ib  = 8u * i + ir;
    if (ib >= pc.nb32) return;

    // Q4_1 block: [fp16 d][fp16 m][qs[16]] = 20 bytes
    uint base     = ib * 20u;
    float d       = unpackHalf2x16(u16_at(base)).x;
    float m       = unpackHalf2x16(u16_at(base + 2u)).x;

    uint q_base   = base + 4u + 4u * il;
    uint out_base = 256u * i + 32u * ir + 4u * il;

    for (uint l = 0u; l < 4u; l++) {
        uint q = byte_at(q_base + l);
        dst.data[out_base + l]        = d * float(q & 0xFu) + m;
        dst.data[out_base + l + 16u]  = d * float(q >> 4u)  + m;
    }
}
