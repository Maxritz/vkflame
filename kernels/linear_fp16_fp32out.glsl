// linear_fp16_fp32out.glsl — FP16 inputs, FP32 output GEMM
// Used by hipblasGemmEx when A/B are fp16 but the output buffer is fp32
// (the common ggml/Ollama pattern: compute in fp16, accumulate/store as fp32).
#version 460
#extension GL_EXT_shader_16bit_storage                     : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) readonly  buffer BufA { float16_t a[]; };
layout(set = 0, binding = 1) readonly  buffer BufB { float16_t b[]; };
layout(set = 0, binding = 2) readonly  buffer BufC { float     c[]; };  // fp32 bias (unused)
layout(set = 0, binding = 3) writeonly buffer BufD { float     d[]; };  // fp32 output

layout(push_constant) uniform PC {
    uint M;
    uint N;
    uint K;
    uint activation;  // 0=none 1=silu 2=relu 3=gelu_erf
} pc;

// Abramowitz & Stegun erf approximation — max error < 1.5e-7
float erf_a(float x) {
    float t = 1.0 / (1.0 + 0.3275911 * abs(x));
    float y = 1.0 - ((((1.061405429*t - 1.453152027)*t + 1.421413741)*t
                       - 0.284496736)*t + 0.254829592)*t * exp(-x*x);
    return (x >= 0.0) ? y : -y;
}

float apply_act(float x) {
    if (pc.activation == 1u) return x * (1.0 / (1.0 + exp(-x)));                     // silu
    if (pc.activation == 2u) return max(0.0, x);                                      // relu
    if (pc.activation == 3u) return 0.5 * x * (1.0 + erf_a(x * 0.70710678118654));  // gelu_erf
    return x;
}

void main() {
    uint row = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;
    uint col = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;

    if (row >= pc.M || col >= pc.N)
        return;

    // Accumulate in fp32 (read fp16 inputs, write fp32 output)
    float acc = 0.0;
    for (uint k = 0u; k < pc.K; k++)
        acc += float(a[row * pc.K + k]) * float(b[k * pc.N + col]);

    d[row * pc.N + col] = apply_act(acc);  // write float32, not float16_t
}
