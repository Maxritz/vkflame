// scale_f32.glsl — element-wise scale + bias: y = scale * x + bias
// Launch: gridDim.x = (n+255)/256, blockDim.x = 256
#version 460
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly  buffer SrcBuf { float data[]; } src;
layout(set = 0, binding = 1) writeonly buffer DstBuf { float data[]; } dst;

layout(push_constant) uniform PC {
    uint  n;
    float scale;
    float bias;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    dst.data[i] = pc.scale * src.data[i] + pc.bias;
}
