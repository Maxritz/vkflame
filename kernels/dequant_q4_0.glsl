// dequant_q4_0.glsl — Q4_0 → float32
// Block format: [fp16 d (2B)][qs[16] (16B)] = 18 bytes, 32 weights per block
// d is global scale; nibble value 0..15, actual = d*(val - 8)
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
    uint il  = tid / 8u;   // sub-group: 0..3
    uint ir  = tid % 8u;   // element within sub-group: 0..7
    uint i   = gl_WorkGroupID.x;
    uint ib  = 8u * i + ir;
    if (ib >= pc.nb32) return;

    // Each Q4_0 block is 18 bytes: [fp16 d][qs[16]]
    uint base     = ib * 18u;
    float d       = unpackHalf2x16(u16_at(base)).x;
    float dm      = -8.0 * d;

    uint q_base   = base + 2u + 4u * il;
    uint out_base = 256u * i + 32u * ir + 4u * il;

    for (uint l = 0u; l < 4u; l++) {
        uint q = byte_at(q_base + l);
        dst.data[out_base + l]        = d * float(q & 0xFu) + dm;
        dst.data[out_base + l + 16u]  = d * float(q >> 4u)  + dm;
    }
}
