#version 460

layout(local_size_x = 64) in;

// Filter weights: [OC, IC, 3, 3] stored flat
layout(set = 0, binding = 0) readonly  buffer Filter { float filter_w[]; };
// Output (transformed): [OC, IC, 4, 4] stored flat
layout(set = 0, binding = 1) writeonly buffer GHat   { float g_hat[]; };

layout(push_constant) uniform PC {
    uint OC;
    uint IC;
} pc;

// G matrix for Winograd F(2,3):
//   G = [[1,    0,    0  ],
//         [0.5,  0.5,  0.5],
//         [0.5, -0.5,  0.5],
//         [0,    0,    1  ]]
// Result: g_hat = G * filter * G^T  (4x4 output from 3x3 input)

void compute_GgGT(uint oc, uint ic) {
    uint f_base = (oc * pc.IC + ic) * 9u; // 3x3 filter

    float g[3][3];
    for (uint r = 0u; r < 3u; r++)
        for (uint c = 0u; c < 3u; c++)
            g[r][c] = filter_w[f_base + r * 3u + c];

    // Tmp = G * g  (4x3 intermediate)
    float tmp[4][3];
    // Row 0: [1, 0, 0] * g
    tmp[0][0] = g[0][0]; tmp[0][1] = g[0][1]; tmp[0][2] = g[0][2];
    // Row 1: [0.5, 0.5, 0.5] * g
    tmp[1][0] = 0.5 * (g[0][0] + g[1][0] + g[2][0]);
    tmp[1][1] = 0.5 * (g[0][1] + g[1][1] + g[2][1]);
    tmp[1][2] = 0.5 * (g[0][2] + g[1][2] + g[2][2]);
    // Row 2: [0.5, -0.5, 0.5] * g
    tmp[2][0] = 0.5 * (g[0][0] - g[1][0] + g[2][0]);
    tmp[2][1] = 0.5 * (g[0][1] - g[1][1] + g[2][1]);
    tmp[2][2] = 0.5 * (g[0][2] - g[1][2] + g[2][2]);
    // Row 3: [0, 0, 1] * g
    tmp[3][0] = g[2][0]; tmp[3][1] = g[2][1]; tmp[3][2] = g[2][2];

    // Result = tmp * G^T  (4x4)
    // G^T columns are: [1, 0, 0, 0]^T not relevant — G^T rows:
    // G^T = [[1, 0.5, 0.5, 0], [0, 0.5, -0.5, 0], [0, 0.5, 0.5, 1]]
    // So Result[i][j] = sum_k tmp[i][k] * G[j][k]  (G is 4x3)
    // G rows: g0=[1,0,0], g1=[0.5,0.5,0.5], g2=[0.5,-0.5,0.5], g3=[0,0,1]
    // Result[i][j] = dot(tmp[i], G[j])

    uint out_base = (oc * pc.IC + ic) * 16u; // 4x4 output
    for (uint i = 0u; i < 4u; i++) {
        // j=0: G[0]=[1,0,0]
        g_hat[out_base + i * 4u + 0u] = tmp[i][0];
        // j=1: G[1]=[0.5,0.5,0.5]
        g_hat[out_base + i * 4u + 1u] = 0.5 * (tmp[i][0] + tmp[i][1] + tmp[i][2]);
        // j=2: G[2]=[0.5,-0.5,0.5]
        g_hat[out_base + i * 4u + 2u] = 0.5 * (tmp[i][0] - tmp[i][1] + tmp[i][2]);
        // j=3: G[3]=[0,0,1]
        g_hat[out_base + i * 4u + 3u] = tmp[i][2];
    }
}

void main() {
    uint global_id = gl_GlobalInvocationID.x;
    uint total = pc.OC * pc.IC;
    if (global_id >= total) return;

    uint oc = global_id / pc.IC;
    uint ic = global_id % pc.IC;

    compute_GgGT(oc, ic);
}
