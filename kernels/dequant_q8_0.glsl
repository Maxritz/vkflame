// dequant_q8_0.glsl — Q8_0 → float32
// Block format: [fp16 d (2B)][int8_t qs[32] (32B)] = 34 bytes, 32 weights per block
// actual = d * qs[i]
// Launch: gridDim.x = nb, blockDim.x = 32 (1 thread per weight in block)
#version 460
layout(local_size_x = 32) in;

layout(set = 0, binding = 0) readonly  buffer InBuf  { uint data[]; } src;
layout(set = 0, binding = 1) writeonly buffer OutBuf { float data[]; } dst;

layout(push_constant) uniform PC { uint nb; } pc;

uint byte_at(uint b) { return (src.data[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu; }
uint u16_at (uint b) { return byte_at(b) | (byte_at(b + 1u) << 8u); }
// Sign-extend 8-bit to int
int sbyte_at(uint b) { int v = int(byte_at(b)); return v - ((v & 128) != 0 ? 256 : 0); }

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint ib  = gl_WorkGroupID.x;
    if (ib >= pc.nb) return;

    // Q8_0 block: [fp16 d][32 × int8] = 34 bytes
    uint  base = ib * 34u;
    float d    = unpackHalf2x16(u16_at(base)).x;
    int   q    = sbyte_at(base + 2u + tid);
    dst.data[ib * 32u + tid] = d * float(q);
}
