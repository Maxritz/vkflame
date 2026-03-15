// dequant_q8_1.glsl — Q8_1 → float32
// Block format: [fp16 d (2B)][fp16 s (2B)][int8_t qs[32] (32B)] = 36 bytes, 32 weights per block
// actual[i] = d * qs[i]
// Used for ggml activation tensors in mul_mat_q*_q8_1 kernels
// Launch: gridDim.x = nb, blockDim.x = 32 (1 thread per weight in block)
#version 460
layout(local_size_x = 32) in;

layout(set = 0, binding = 0) readonly  buffer InBuf  { uint data[]; } src;
layout(set = 0, binding = 1) writeonly buffer OutBuf { float data[]; } dst;

layout(push_constant) uniform PC { uint nb; } pc;

// Read a single byte from a uint-packed buffer
uint byte_at(uint b) { return (src.data[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu; }
// Read a little-endian fp16 from byte offset b
uint u16_at(uint b) { return byte_at(b) | (byte_at(b + 1u) << 8u); }
// Sign-extend int8 to int
int sbyte_at(uint b) { int v = int(byte_at(b)); return v - ((v & 128) != 0 ? 256 : 0); }

void main() {
    uint tid = gl_LocalInvocationID.x;  // 0..31 — one thread per element in block
    uint ib  = gl_WorkGroupID.x;
    if (ib >= pc.nb) return;

    // Q8_1 block: [fp16 d (2B)][fp16 s (2B)][32 × int8_t qs] = 36 bytes total
    uint  base = ib * 36u;
    float d    = unpackHalf2x16(u16_at(base)).x;       // delta (scale)
    // s at base+2 is the sum; not needed for element-wise decode
    int   q    = sbyte_at(base + 4u + tid);             // int8 quantised value
    dst.data[ib * 32u + tid] = d * float(q);
}
