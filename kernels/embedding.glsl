#version 460
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly  buffer WeightBuf  { float16_t weight[]; };  // [V, D]
layout(set = 0, binding = 1) readonly  buffer IndexBuf   { int indices[]; };        // [B]
layout(set = 0, binding = 2) writeonly buffer OutBuf     { float16_t out_data[]; }; // [B, D]

layout(push_constant) uniform PC {
    uint V;  // vocabulary size
    uint D;  // embedding dimension
    uint B;  // batch / number of lookup indices
} pc;

void main() {
    // Total elements = B * D; each thread handles one element
    uint global_id = gl_GlobalInvocationID.x;
    uint total = pc.B * pc.D;
    if (global_id >= total) return;

    uint b   = global_id / pc.D;
    uint dim = global_id % pc.D;

    int  idx = indices[b];
    // Bounds check: out-of-range index → write 0.0
    if (idx < 0 || uint(idx) >= pc.V) {
        out_data[b * pc.D + dim] = float16_t(0.0);
        return;
    }

    out_data[b * pc.D + dim] = weight[uint(idx) * pc.D + dim];
}
