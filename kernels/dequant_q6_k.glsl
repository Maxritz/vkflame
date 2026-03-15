// dequant_q6_k.glsl — Q6_K → float32
// Block: [ql[128] (128B)][qh[64] (64B)][scales[16] (16B)][fp16 d (2B)] = 210 bytes, 256 weights
// Each weight uses 6 bits: 4 from ql nibbles + 2 from qh
// Launch: gridDim.x = nb (one workgroup per block), blockDim.x = 64
#version 460
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly  buffer InBuf  { uint data[]; } src;
layout(set = 0, binding = 1) writeonly buffer OutBuf { float data[]; } dst;

layout(push_constant) uniform PC { uint nb; } pc;

uint byte_at(uint b) { return (src.data[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu; }
uint u16_at (uint b) { return byte_at(b) | (byte_at(b + 1u) << 8u); }
int  sbyte_at(uint b) { int v = int(byte_at(b)); return v - ((v & 128) != 0 ? 256 : 0); }

void main() {
    uint tid = gl_LocalInvocationID.x;   // 0..63
    uint ip  = tid / 32u;                // 0 or 1 (half of block)
    uint il  = tid - 32u * ip;           // 0..31 within half
    uint is_scale = 8u * ip + il / 16u;  // scale index within block
    uint ib  = gl_WorkGroupID.x;
    if (ib >= pc.nb) return;

    // Q6_K layout (210 bytes):
    // [ql[128]][qh[64]][scales[16]][fp16 d (2)]
    // offsets: 0       128          192         208
    uint base       = ib * 210u;
    uint ql_off     = base + 0u;
    uint qh_off     = base + 128u;
    uint scales_off = base + 192u;
    float d         = unpackHalf2x16(u16_at(base + 208u)).x;

    // ql pointer: ql + 64*ip + il
    uint ql_ptr = ql_off + 64u * ip + il;
    // qh pointer: qh + 32*ip + il
    uint qh_b = byte_at(qh_off + 32u * ip + il);

    // Signed scales: int8_t scales[is_scale] and scales[is_scale+2] etc.
    int sc0 = sbyte_at(scales_off + is_scale + 0u);
    int sc2 = sbyte_at(scales_off + is_scale + 2u);
    int sc4 = sbyte_at(scales_off + is_scale + 4u);
    int sc6 = sbyte_at(scales_off + is_scale + 6u);

    uint ql0  = byte_at(ql_ptr);
    uint ql32 = byte_at(ql_ptr + 32u);

    // Reconstruct 6-bit signed values (centered at 32, range -32..31)
    int v0  = int((ql0  & 0xFu) | (((qh_b >> 0u) & 3u) << 4u)) - 32;
    int v32 = int((ql32 & 0xFu) | (((qh_b >> 2u) & 3u) << 4u)) - 32;
    int v64 = int((ql0  >> 4u)  | (((qh_b >> 4u) & 3u) << 4u)) - 32;
    int v96 = int((ql32 >> 4u)  | (((qh_b >> 6u) & 3u) << 4u)) - 32;

    uint out_base = ib * 256u + 128u * ip + il;
    dst.data[out_base + 0u]  = d * float(sc0 * v0);
    dst.data[out_base + 32u] = d * float(sc2 * v32);
    dst.data[out_base + 64u] = d * float(sc4 * v64);
    dst.data[out_base + 96u] = d * float(sc6 * v96);
}
