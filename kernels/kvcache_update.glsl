#version 460
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 256) in;

// KV cache: [max_seq, n_heads, head_dim] — read-write
layout(set = 0, binding = 0) buffer CacheBuf  { float16_t cache[]; };
// New KV slice to write: [n_heads, head_dim]
layout(set = 0, binding = 1) readonly buffer NewKV { float16_t new_kv[]; };

layout(push_constant) uniform PC {
    uint seq_pos;   // which sequence position to write into
    uint n_heads;
    uint head_dim;
    uint max_seq;
} pc;

void main() {
    // Total elements = n_heads * head_dim
    uint global_id = gl_GlobalInvocationID.x;
    uint total = pc.n_heads * pc.head_dim;
    if (global_id >= total) return;

    uint head = global_id / pc.head_dim;
    uint d    = global_id % pc.head_dim;

    // Destination in cache:  cache[seq_pos * n_heads * head_dim + head * head_dim + d]
    uint dst = pc.seq_pos * pc.n_heads * pc.head_dim + head * pc.head_dim + d;
    // Source in new_kv:      new_kv[head * head_dim + d]
    uint src = head * pc.head_dim + d;

    cache[dst] = new_kv[src];
}
