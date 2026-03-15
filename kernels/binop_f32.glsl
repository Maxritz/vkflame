// binop_f32.glsl — element-wise binary ops on float32: add, mul, sub, div
// src1 is broadcast-aware: src1[i % src1_n]
// Launch: gridDim.x = (n+255)/256, blockDim.x = 256
#version 460
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly  buffer Src0Buf { float data[]; } src0;
layout(set = 0, binding = 1) readonly  buffer Src1Buf { float data[]; } src1;
layout(set = 0, binding = 2) writeonly buffer DstBuf  { float data[]; } dst;

layout(push_constant) uniform PC {
    uint n;        // number of output elements
    uint op;       // 0=ADD 1=MUL 2=SUB 3=DIV
    uint src1_n;   // size of src1 (for broadcast: i % src1_n)
    uint pad;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;

    float a = src0.data[i];
    float b = src1.data[i % pc.src1_n];

    float y;
    switch (pc.op) {
        case 0u:  y = a + b; break;
        case 1u:  y = a * b; break;
        case 2u:  y = a - b; break;
        case 3u:  y = (b != 0.0) ? a / b : 0.0; break;
        default:  y = a;
    }
    dst.data[i] = y;
}
