#version 460

layout(local_size_x = 4, local_size_y = 4) in;

// Input:  [N, C, H, W]
layout(set = 0, binding = 0) readonly  buffer Input  { float input_data[]; };
// Filter transformed: [OC, IC, 4, 4] from winograd_filter_transform
layout(set = 0, binding = 1) readonly  buffer FilterT{ float g_hat[]; };
// Output: [N, OC, Ho, Wo]  where Ho = (H - 2), Wo = (W - 2)
layout(set = 0, binding = 2) writeonly buffer Output { float output_data[]; };

layout(push_constant) uniform PC {
    uint N;    // batch
    uint C;    // input channels
    uint OC;   // output channels
    uint H;    // input height
    uint W;    // input width
    uint Ho;   // output height = H - 2
    uint Wo;   // output width  = W - 2
} pc;

// Bt matrix for input transform (Winograd F(2,3)):
// Bt = [[ 1,  0, -1,  0],
//        [ 0,  1,  1,  0],
//        [ 0, -1,  1,  0],
//        [ 0,  1,  0, -1]]
//
// At matrix for output transform:
// At = [[1, 1, 1, 0],
//        [0, 1,-1,-1]]

// One workgroup per (n, oc, tile_row, tile_col) for each 2x2 output tile
// WorkGroupID: x=tile_col_idx, y=tile_row_idx, z=(n*OC + oc)
void main() {
    uint n_oc   = gl_WorkGroupID.z;
    uint n      = n_oc / pc.OC;
    uint oc     = n_oc % pc.OC;
    uint tile_r = gl_WorkGroupID.y;
    uint tile_c = gl_WorkGroupID.x;
    uint tr     = gl_LocalInvocationID.y;  // 0..3
    uint tc     = gl_LocalInvocationID.x;  // 0..3

    // Top-left corner of this 4x4 input patch (2x2 output tile)
    uint in_r0 = tile_r * 2u;
    uint in_c0 = tile_c * 2u;

    // Accumulate over input channels
    float acc[4][4];
    for (uint i = 0u; i < 4u; i++)
        for (uint j = 0u; j < 4u; j++)
            acc[i][j] = 0.0;

    for (uint ic = 0u; ic < pc.C; ic++) {
        // Load 4x4 input tile into registers d[4][4]
        float d[4][4];
        for (uint r = 0u; r < 4u; r++) {
            for (uint c = 0u; c < 4u; c++) {
                uint in_r = in_r0 + r;
                uint in_c = in_c0 + c;
                if (in_r < pc.H && in_c < pc.W) {
                    uint idx = ((n * pc.C + ic) * pc.H + in_r) * pc.W + in_c;
                    d[r][c] = input_data[idx];
                } else {
                    d[r][c] = 0.0;
                }
            }
        }

        // Input transform: v = Bt * d * B
        // Bt * d  (4x4 * 4x4 = 4x4, but B = Bt^T)
        // B   = [[ 1, 0, 0,  0],
        //         [ 0, 1,-1,  1],
        //         [-1, 1, 1,  0],
        //         [ 0, 0, 0, -1]]
        // Apply Bt row-wise, then B column-wise

        float tmp[4][4];
        // tmp = Bt * d  (apply Bt to rows)
        for (uint c = 0u; c < 4u; c++) {
            tmp[0][c] =  d[0][c] - d[2][c];           // Bt[0] = [1, 0,-1, 0]
            tmp[1][c] =  d[1][c] + d[2][c];           // Bt[1] = [0, 1, 1, 0]
            tmp[2][c] = -d[1][c] + d[2][c];           // Bt[2] = [0,-1, 1, 0]
            tmp[3][c] =  d[1][c] - d[3][c];           // Bt[3] = [0, 1, 0,-1]
        }

        float v[4][4];
        // v = tmp * B  (apply B to columns)
        for (uint r = 0u; r < 4u; r++) {
            v[r][0] =  tmp[r][0] - tmp[r][2];         // B[0] = [1, 0,-1, 0]^T
            v[r][1] =  tmp[r][1] + tmp[r][2];         // B[1] = [0, 1, 1, 0]^T
            v[r][2] = -tmp[r][1] + tmp[r][2];         // B[2] = [0,-1, 1, 0]^T
            v[r][3] =  tmp[r][1] - tmp[r][3];         // B[3] = [0, 1, 0,-1]^T
        }

        // Load transformed filter g_hat[oc, ic, :, :] (4x4)
        uint ghat_base = (oc * pc.C + ic) * 16u;
        float gh[4][4];
        for (uint r = 0u; r < 4u; r++)
            for (uint c = 0u; c < 4u; c++)
                gh[r][c] = g_hat[ghat_base + r * 4u + c];

        // Element-wise multiply in Winograd domain
        for (uint r = 0u; r < 4u; r++)
            for (uint c = 0u; c < 4u; c++)
                acc[r][c] += v[r][c] * gh[r][c];
    }

    // Output transform: m = At * acc * A
    // At = [[1, 1, 1, 0],
    //        [0, 1,-1,-1]]
    // A  = [[1, 0],
    //        [1, 1],
    //        [1,-1],
    //        [0,-1]]

    float tmp2[2][4];
    // At * acc  (2x4 result)
    for (uint c = 0u; c < 4u; c++) {
        tmp2[0][c] =  acc[0][c] + acc[1][c] + acc[2][c];  // At[0] = [1,1,1,0]
        tmp2[1][c] =  acc[1][c] - acc[2][c] - acc[3][c];  // At[1] = [0,1,-1,-1]
    }

    float m[2][2];
    // tmp2 * A  (2x2 result)
    for (uint r = 0u; r < 2u; r++) {
        m[r][0] =  tmp2[r][0] + tmp2[r][1] + tmp2[r][2];  // A[0] = [1,1,1,0]^T
        m[r][1] =  tmp2[r][1] - tmp2[r][2] - tmp2[r][3];  // A[1] = [0,1,-1,-1]^T
    }

    // Write 2x2 output tile
    for (uint r = 0u; r < 2u; r++) {
        for (uint c = 0u; c < 2u; c++) {
            uint out_r = tile_r * 2u + r;
            uint out_c = tile_c * 2u + c;
            if (out_r < pc.Ho && out_c < pc.Wo) {
                uint out_idx = ((n * pc.OC + oc) * pc.Ho + out_r) * pc.Wo + out_c;
                output_data[out_idx] = m[r][c];
            }
        }
    }
}
