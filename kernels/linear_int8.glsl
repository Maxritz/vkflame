#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int8  : require
#extension GL_EXT_shader_16bit_storage                   : require

layout(local_size_x = 16, local_size_y = 16) in;

// A: input activations [M x K] in fp16
layout(set = 0, binding = 0) readonly buffer A       { float16_t a[]; };
// B: weight matrix [N x K] in int8 (row-major, weights already quantised)
layout(set = 0, binding = 1) readonly buffer B       { int8_t b[]; };
// C: bias [N] (optional, may be empty)
layout(set = 0, binding = 2) readonly buffer C       { float c[]; };
// W_scale: per-output-channel dequant scale [N]
layout(set = 0, binding = 3) readonly buffer WScale  { float w_scale[]; };
// D: output [M x N] in fp16
layout(set = 0, binding = 4) writeonly buffer D      { float16_t d_out[]; };

layout(push_constant) uniform PC {
    uint  M;
    uint  N;
    uint  K;
    float act_scale;   // input quantisation scale
    uint  activation;  // 0=none, 1=silu, 2=relu, 3=gelu_erf
} pc;

// SiLU: x * sigmoid(x)
float silu(float x) {
    return x / (1.0 + exp(-x));
}

// GELU erf form (NOT tanh approximation)
// erf approximation: Abramowitz & Stegun 7.1.26, max error 1.5e-7
float erf_approx(float x) {
    float t = 1.0 / (1.0 + 0.3275911 * abs(x));
    float poly = t * (0.254829592
                 + t * (-0.284496736
                 + t * (1.421413741
                 + t * (-1.453152027
                 + t * 1.061405429))));
    float val = 1.0 - poly * exp(-x * x);
    return sign(x) * val;
}

float gelu_erf(float x) {
    return 0.5 * x * (1.0 + erf_approx(x * 0.70710678));
}

void main() {
    uint m = gl_GlobalInvocationID.y;  // output row
    uint n = gl_GlobalInvocationID.x;  // output column

    if (m >= pc.M || n >= pc.N) return;

    int acc = 0;

    // Process K in steps of 4 (dot product of i8vec4)
    uint k4 = pc.K / 4u;
    for (uint ki = 0u; ki < k4; ki++) {
        uint k_base = ki * 4u;

        // Quantise 4 input activations from fp16 to int8
        float x0 = float(a[m * pc.K + k_base + 0u]);
        float x1 = float(a[m * pc.K + k_base + 1u]);
        float x2 = float(a[m * pc.K + k_base + 2u]);
        float x3 = float(a[m * pc.K + k_base + 3u]);

        int8_t xq0 = int8_t(clamp(int(round(x0 / pc.act_scale)), -127, 127));
        int8_t xq1 = int8_t(clamp(int(round(x1 / pc.act_scale)), -127, 127));
        int8_t xq2 = int8_t(clamp(int(round(x2 / pc.act_scale)), -127, 127));
        int8_t xq3 = int8_t(clamp(int(round(x3 / pc.act_scale)), -127, 127));

        // Load 4 weight bytes for output neuron n
        int8_t w0 = b[n * pc.K + k_base + 0u];
        int8_t w1 = b[n * pc.K + k_base + 1u];
        int8_t w2 = b[n * pc.K + k_base + 2u];
        int8_t w3 = b[n * pc.K + k_base + 3u];

        // Integer dot product via manual int32 multiply-accumulate
        // (compiler will map to native IDOT instructions where hardware supports it)
        acc += int(xq0) * int(w0) + int(xq1) * int(w1)
             + int(xq2) * int(w2) + int(xq3) * int(w3);
    }

    // Handle remaining K elements (K % 4 != 0)
    for (uint ki = k4 * 4u; ki < pc.K; ki++) {
        float xf = float(a[m * pc.K + ki]);
        int8_t xq = int8_t(clamp(int(round(xf / pc.act_scale)), -127, 127));
        int8_t wq = b[n * pc.K + ki];
        acc += int(xq) * int(wq);
    }

    // Dequantise: multiply by per-channel weight scale
    float result = float(acc) * w_scale[n];

    // Optional activation function
    if      (pc.activation == 1u) result = silu(result);
    else if (pc.activation == 2u) result = max(0.0, result);
    else if (pc.activation == 3u) result = gelu_erf(result);

    d_out[m * pc.N + n] = float16_t(result);
}
