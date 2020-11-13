#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_AMD_gpu_shader_half_float: enable

layout(binding=0) buffer InputA { uvec4 x[]; } inputA;
layout(binding=1) buffer InputB { uvec4 x[]; } inputB;
layout(binding=2) buffer Output { uvec4 x[]; } outputO;
layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint M = 1;
layout(constant_id = 1) const uint N = 1;
layout(constant_id = 2) const uint K = 1;

const uint strideA = K;
const uint strideB = N;
const uint strideC = N;

const uint C_ROWS = TILE_M / 1;
const uint C_COLS = TILE_N / 128;

uint coordToOffset(uint i, uint j, uint stride)
{
    return (stride * i + j);
}

void main()
{
    uint gID = gl_WorkGroupID.x;
    uint laneId = gl_LocalInvocationID.x;
    uvec2 tileID = uvec2(gl_GlobalInvocationID.xy);
    f16vec2 C[C_ROWS][C_COLS][4];
    uvec4 B[8][C_COLS];

    // Initialize result to zero
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
          [[unroll]] for (uint k = 0; j < 4; ++k) {
            C[i][j][k] = f16vec2(0, 0);
          }
        }
    }
    for (uint k = 0; k < K; k+=8) {
      [[unroll]] for (uint i = 0; i < 8; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
          uint gj = gID * (TILE_N / 8) + laneId +j*16;
          uint gk = k+i;
          B[i][j] = inputB.x[coordToOffset(gk, gj, strideB/8)];
        }
      }
      
      [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        uint gi = tileID.y*C_ROWS+i;
        uint gk = k/8;
        uvec4 A = inputA.x[coordToOffset(gi, gk, strideA/8)];
        f16vec2 a;
        f16vec2 b;
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
          a = unpackFloat2x16(A.x);
          b = unpackFloat2x16(B[0][j].x);
          C[i][j][0] += f16vec2(a.x, a.x) * b;
          b = unpackFloat2x16(B[0][j].y);
          C[i][j][1] += f16vec2(a.x, a.x) * b;
          b = unpackFloat2x16(B[0][j].z);
          C[i][j][2] += f16vec2(a.x, a.x) * b;
          b = unpackFloat2x16(B[0][j].w);
          C[i][j][3] += f16vec2(a.x, a.x) * b;

          b = unpackFloat2x16(B[1][j].x);
          C[i][j][0] += f16vec2(a.y, a.y) * b;
          b = unpackFloat2x16(B[1][j].y);
          C[i][j][1] += f16vec2(a.y, a.y) * b;
          b = unpackFloat2x16(B[1][j].z);
          C[i][j][2] += f16vec2(a.y, a.y) * b;
          b = unpackFloat2x16(B[1][j].w);
          C[i][j][3] += f16vec2(a.y, a.y) * b;

          a = unpackFloat2x16(A.y);
          b = unpackFloat2x16(B[2][j].x);
          C[i][j][0] += f16vec2(a.x, a.x) * b;
          b = unpackFloat2x16(B[2][j].y);
          C[i][j][1] += f16vec2(a.x, a.x) * b;
          b = unpackFloat2x16(B[2][j].z);
          C[i][j][2] += f16vec2(a.x, a.x) * b;
          b = unpackFloat2x16(B[2][j].w);
          C[i][j][3] += f16vec2(a.x, a.x) * b;

          b = unpackFloat2x16(B[3][j].x);
          C[i][j][0] += f16vec2(a.y, a.y) * b;
          b = unpackFloat2x16(B[3][j].y);
          C[i][j][1] += f16vec2(a.y, a.y) * b;
          b = unpackFloat2x16(B[3][j].z);
          C[i][j][2] += f16vec2(a.y, a.y) * b;
          b = unpackFloat2x16(B[3][j].w);
          C[i][j][3] += f16vec2(a.y, a.y) * b;

          a = unpackFloat2x16(A.z);
          b = unpackFloat2x16(B[4][j].x);
          C[i][j][0] += f16vec2(a.x, a.x) * b;
          b = unpackFloat2x16(B[4][j].y);
          C[i][j][1] += f16vec2(a.x, a.x) * b;
          b = unpackFloat2x16(B[4][j].z);
          C[i][j][2] += f16vec2(a.x, a.x) * b;
          b = unpackFloat2x16(B[4][j].w);
          C[i][j][3] += f16vec2(a.x, a.x) * b;

          b = unpackFloat2x16(B[5][j].x);
          C[i][j][0] += f16vec2(a.y, a.y) * b;
          b = unpackFloat2x16(B[5][j].y);
          C[i][j][1] += f16vec2(a.y, a.y) * b;
          b = unpackFloat2x16(B[5][j].z);
          C[i][j][2] += f16vec2(a.y, a.y) * b;
          b = unpackFloat2x16(B[5][j].w);
          C[i][j][3] += f16vec2(a.y, a.y) * b;

          a = unpackFloat2x16(A.w);
          b = unpackFloat2x16(B[6][j].x);
          C[i][j][0] += f16vec2(a.x, a.x) * b;
          b = unpackFloat2x16(B[6][j].y);
          C[i][j][1] += f16vec2(a.x, a.x) * b;
          b = unpackFloat2x16(B[6][j].z);
          C[i][j][2] += f16vec2(a.x, a.x) * b;
          b = unpackFloat2x16(B[6][j].w);
          C[i][j][3] += f16vec2(a.x, a.x) * b;

          b = unpackFloat2x16(B[7][j].x);
          C[i][j][0] += f16vec2(a.y, a.y) * b;
          b = unpackFloat2x16(B[7][j].y);
          C[i][j][1] += f16vec2(a.y, a.y) * b;
          b = unpackFloat2x16(B[7][j].z);
          C[i][j][2] += f16vec2(a.y, a.y) * b;
          b = unpackFloat2x16(B[7][j].w);
          C[i][j][3] += f16vec2(a.y, a.y) * b;
        }
      }
    }

    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gi = tileID.y*C_ROWS+i;
            uint gj = gID * (TILE_N / 8) + laneId +j*16;
            uvec4 v = uvec4(packFloat2x16(C[i][j][0]), 
                            packFloat2x16(C[i][j][1]), 
                            packFloat2x16(C[i][j][2]), 
                            packFloat2x16(C[i][j][3]));
            outputO.x[gi * strideC/8 + gj] = v;
        }
    }
}