// linear_coop.glsl — FP16 GEMM using VK_KHR_cooperative_matrix  (WMMA on RDNA4)
// Used for large matmuls where M*N*K > 2^30; one wave64 subgroup per 16x16 tile.
#version 460
#extension GL_KHR_cooperative_matrix                       : require
#extension GL_EXT_shader_16bit_storage                     : require
#extension GL_EXT_shader_explicit_arithmetic_types         : require
#extension GL_KHR_memory_scope_semantics                   : require
#extension GL_KHR_shader_subgroup_basic                    : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly  buffer BufA { float16_t a[]; };
layout(set = 0, binding = 1) readonly  buffer BufB { float16_t b[]; };
layout(set = 0, binding = 2) readonly  buffer BufC { float16_t c[]; };  // unused
layout(set = 0, binding = 3) writeonly buffer BufD { float16_t d[]; };

layout(push_constant) uniform PC {
    uint M;
    uint N;
    uint K;
    uint activation;  // 0=none 1=silu 2=relu 3=gelu_erf
} pc;

// Temporary storage for writing fp32 accumulator back as fp16
shared float s_temp[16 * 16];

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
    uint tileRow = gl_WorkGroupID.y * 16u;
    uint tileCol = gl_WorkGroupID.x * 16u;

    // Accumulator in fp32 for numerical stability
    coopmat<float32_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> matC;
    matC = coopmat<float32_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    uint numTiles = (pc.K + 15u) / 16u;
    for (uint t = 0u; t < numTiles; t++) {
        uint kBase = t * 16u;

        // Load A tile [tileRow..tileRow+16, kBase..kBase+16], row-major stride = K
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
        coopMatLoad(matA, a, tileRow * pc.K + kBase, pc.K,
                    gl_CooperativeMatrixLayoutRowMajor);

        // Load B tile [kBase..kBase+16, tileCol..tileCol+16], row-major stride = N
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
        coopMatLoad(matB, b, kBase * pc.N + tileCol, pc.N,
                    gl_CooperativeMatrixLayoutRowMajor);

        matC = coopMatMulAdd(matA, matB, matC);
    }

    // Apply activation element-wise — 16x16 tile / 64 threads = 4 elements per invocation
    if (pc.activation != 0u) {
        for (uint i = 0u; i < 4u; i++)
            matC[i] = apply_act(matC[i]);
    }

    // Write fp32 accumulator to shared memory, then store as fp16
    coopMatStore(matC, s_temp, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
    barrier();

    // 64 threads write 4 elements each = 256 = 16*16
    uint lid = gl_LocalInvocationID.x;
    for (uint i = 0u; i < 4u; i++) {
        uint idx = lid * 4u + i;
        uint r   = idx / 16u;
        uint c   = idx % 16u;
        if ((tileRow + r) < pc.M && (tileCol + c) < pc.N)
            d[(tileRow + r) * pc.N + (tileCol + c)] = float16_t(s_temp[idx]);
    }
}
