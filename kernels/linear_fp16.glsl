// linear_fp16.glsl — FP16 GEMM  D[M x N] = activation(A[M x K] * B[K x N])
// Simple direct accumulation — no shared memory, one thread per output element.
// Avoids shared-mem barrier races; correct on all RDNA drivers.
#version 460
#extension GL_EXT_shader_16bit_storage                     : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) readonly  buffer BufA { float16_t a[]; };
layout(set = 0, binding = 1) readonly  buffer BufB { float16_t b[]; };
layout(set = 0, binding = 2) readonly  buffer BufC { float16_t c[]; };  // unused / bias
layout(set = 0, binding = 3) writeonly buffer BufD { float16_t d[]; };

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

// Activation functions (GELU uses erf, not tanh — see AGENTS.md)
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

    // Direct accumulation in fp32 — no shared mem, no barrier needed
    float acc = 0.0;
    for (uint k = 0u; k < pc.K; k++)
        acc += float(a[row * pc.K + k]) * float(b[k * pc.N + col]);

    d[row * pc.N + col] = float16_t(apply_act(acc));
}
