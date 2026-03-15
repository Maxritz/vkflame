// elementwise_f32.glsl — element-wise unary ops on float32
// Covers: silu, gelu_erf, relu, tanh, gelu_quick, neg, sigmoid, sqr, sqrt, exp, log, abs, sgn
// Launch: gridDim.x = (n+255)/256, blockDim.x = 256
#version 460
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly  buffer SrcBuf { float data[]; } src;
layout(set = 0, binding = 1) writeonly buffer DstBuf { float data[]; } dst;

layout(push_constant) uniform PC {
    uint n;    // number of elements
    uint op;   // operation ID (see defines below)
} pc;

// op IDs match VKF_EW_* in dispatch.h
// 0=SILU  1=GELU_ERF  2=RELU  3=TANH  4=GELU_QUICK  5=NEG  6=SIGMOID
// 7=SQR   8=SQRT      9=EXP  10=LOG  11=ABS         12=SGN 13=HARDSIGMOID 14=HARDSWISH

const float SQRT2_INV = 0.7071067811865476;

// GLSL has no erf() — Abramowitz & Stegun 7.1.26 polynomial approximation (max error 1.5e-7)
float erf_approx(float x) {
    float t = 1.0 / (1.0 + 0.3275911 * abs(x));
    float p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
              + t * (-1.453152027 + t * 1.061405429))));
    return sign(x) * (1.0 - p * exp(-x * x));
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;

    float x = src.data[i];
    float y;

    switch (pc.op) {
        case 0u:  y = x / (1.0 + exp(-x));                                   break; // silu
        case 1u:  y = 0.5 * x * (1.0 + erf_approx(x * SQRT2_INV));                 break; // gelu_erf
        case 2u:  y = max(x, 0.0);                                           break; // relu
        case 3u:  y = tanh(x);                                               break; // tanh
        case 4u:  y = x * (1.0 / (1.0 + exp(-1.702 * x)));                  break; // gelu_quick
        case 5u:  y = -x;                                                    break; // neg
        case 6u:  y = 1.0 / (1.0 + exp(-x));                                break; // sigmoid
        case 7u:  y = x * x;                                                 break; // sqr
        case 8u:  y = sqrt(x);                                               break; // sqrt
        case 9u:  y = exp(x);                                                break; // exp
        case 10u: y = log(x);                                                break; // log
        case 11u: y = abs(x);                                                break; // abs
        case 12u: y = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);         break; // sgn
        case 13u: y = clamp((x + 3.0) / 6.0, 0.0, 1.0);                    break; // hardsigmoid
        case 14u: y = x * clamp((x + 3.0) / 6.0, 0.0, 1.0);               break; // hardswish
        default:  y = x;                                                     break;
    }
    dst.data[i] = y;
}
