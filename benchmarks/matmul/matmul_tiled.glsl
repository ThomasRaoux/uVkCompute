#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_EXT_control_flow_attributes : enable

const uint subgroupsize = 16;

layout(binding=0) buffer InputA { vec4 x[]; } inputA;
layout(binding=1) buffer InputB { vec4 x[]; } inputB;
layout(binding=2) buffer Output { vec4 x[]; } outputO;
layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint M = 1;
layout(constant_id = 1) const uint N = 1;
layout(constant_id = 2) const uint K = 1;

const uint strideA = K;
const uint strideB = N;
const uint strideC = N;

const uint lM = subgroupsize;
const uint lN = subgroupsize;
const uint lK = 4;

const uint C_ROWS = TILE_M / subgroupsize;
const uint C_COLS = TILE_N / subgroupsize;

uint coordToOffset(uint i, uint j, uint stride)
{
    return (stride * i + j);
}

void matmul16x16x4(inout vec4 C0, inout vec4 C1, inout vec4 C2, inout vec4 C3, vec4 A, vec4 B) {
  C0.x += A.x * subgroupBroadcast(B.x, 0);
  C0.x += A.y * subgroupBroadcast(B.x, 4);
  C0.x += A.z * subgroupBroadcast(B.x, 8);
  C0.x += A.w * subgroupBroadcast(B.x, 12);
  C0.y += A.x * subgroupBroadcast(B.y, 0);
  C0.y += A.y * subgroupBroadcast(B.y, 4);
  C0.y += A.z * subgroupBroadcast(B.y, 8);
  C0.y += A.w * subgroupBroadcast(B.y, 12);
  C0.z += A.x * subgroupBroadcast(B.z, 0);
  C0.z += A.y * subgroupBroadcast(B.z, 4);
  C0.z += A.z * subgroupBroadcast(B.z, 8);
  C0.z += A.w * subgroupBroadcast(B.z, 12);
  C0.w += A.x * subgroupBroadcast(B.w, 0);
  C0.w += A.y * subgroupBroadcast(B.w, 4);
  C0.w += A.z * subgroupBroadcast(B.w, 8);
  C0.w += A.w * subgroupBroadcast(B.w, 12);
  C1.x += A.x * subgroupBroadcast(B.x, 1);
  C1.x += A.y * subgroupBroadcast(B.x, 5);
  C1.x += A.z * subgroupBroadcast(B.x, 9);
  C1.x += A.w * subgroupBroadcast(B.x, 13);
  C1.y += A.x * subgroupBroadcast(B.y, 1);
  C1.y += A.y * subgroupBroadcast(B.y, 5);
  C1.y += A.z * subgroupBroadcast(B.y, 9);
  C1.y += A.w * subgroupBroadcast(B.y, 13);
  C1.z += A.x * subgroupBroadcast(B.z, 1);
  C1.z += A.y * subgroupBroadcast(B.z, 5);
  C1.z += A.z * subgroupBroadcast(B.z, 9);
  C1.z += A.w * subgroupBroadcast(B.z, 13);
  C1.w += A.x * subgroupBroadcast(B.w, 1);
  C1.w += A.y * subgroupBroadcast(B.w, 5);
  C1.w += A.z * subgroupBroadcast(B.w, 9);
  C1.w += A.w * subgroupBroadcast(B.w, 13);

  C2.x += A.x * subgroupBroadcast(B.x, 2);
  C2.x += A.y * subgroupBroadcast(B.x, 6);
  C2.x += A.z * subgroupBroadcast(B.x, 10);
  C2.x += A.w * subgroupBroadcast(B.x, 14);
  C2.y += A.x * subgroupBroadcast(B.y, 2);
  C2.y += A.y * subgroupBroadcast(B.y, 6);
  C2.y += A.z * subgroupBroadcast(B.y, 10);
  C2.y += A.w * subgroupBroadcast(B.y, 14);
  C2.z += A.x * subgroupBroadcast(B.z, 2);
  C2.z += A.y * subgroupBroadcast(B.z, 6);
  C2.z += A.z * subgroupBroadcast(B.z, 10);
  C2.z += A.w * subgroupBroadcast(B.z, 14);
  C2.w += A.x * subgroupBroadcast(B.w, 2);
  C2.w += A.y * subgroupBroadcast(B.w, 6);
  C2.w += A.z * subgroupBroadcast(B.w, 10);
  C2.w += A.w * subgroupBroadcast(B.w, 14);
  C3.x += A.x * subgroupBroadcast(B.x, 3);
  C3.x += A.y * subgroupBroadcast(B.x, 7);
  C3.x += A.z * subgroupBroadcast(B.x, 11);
  C3.x += A.w * subgroupBroadcast(B.x, 15);
  C3.y += A.x * subgroupBroadcast(B.y, 3);
  C3.y += A.y * subgroupBroadcast(B.y, 7);
  C3.y += A.z * subgroupBroadcast(B.y, 11);
  C3.y += A.w * subgroupBroadcast(B.y, 15);
  C3.z += A.x * subgroupBroadcast(B.z, 3);
  C3.z += A.y * subgroupBroadcast(B.z, 7);
  C3.z += A.z * subgroupBroadcast(B.z, 11);
  C3.z += A.w * subgroupBroadcast(B.z, 15);
  C3.w += A.x * subgroupBroadcast(B.w, 3);
  C3.w += A.y * subgroupBroadcast(B.w, 7);
  C3.w += A.z * subgroupBroadcast(B.w, 11);
  C3.w += A.w * subgroupBroadcast(B.w, 15);
}

void main()
{
    vec4 C0[C_ROWS][C_COLS];
    vec4 C1[C_ROWS][C_COLS];
    vec4 C2[C_ROWS][C_COLS];
    vec4 C3[C_ROWS][C_COLS];
    uvec2 tileID = uvec2(gl_WorkGroupID.xy);

    // Initialize result to zero
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            C0[i][j] = vec4(0.f, 0.f, 0.f, 0.f);
            C1[i][j] = vec4(0.f, 0.f, 0.f, 0.f);
            C2[i][j] = vec4(0.f, 0.f, 0.f, 0.f);
            C3[i][j] = vec4(0.f, 0.f, 0.f, 0.f);
        }
    }

    uint laneId = gl_LocalInvocationID.x;
    for (uint chunkK = 0; chunkK < K; chunkK += lK) {
        vec4 matA[C_ROWS];
        [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
            uint gi = TILE_M * tileID.y + lM * i;
            uint gk = chunkK;
            matA[i] = inputA.x[(coordToOffset(gi, gk, strideA)+strideA*laneId)/4];
        }
        vec4 matB;
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gj = TILE_N * tileID.x + lN * j;
            uint gk = chunkK;
            matB = inputB.x[coordToOffset(gk, gj, strideB)/4 + ((laneId*4)%subgroupsize)/4 + (laneId*4)/subgroupsize*strideB/4];
            [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
                matmul16x16x4(C0[i][j], C1[i][j], C2[i][j], C3[i][j], matA[i], matB);
            }
        }
    }
        
    // Initialize result to zero
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gi = TILE_M * tileID.y + lM * i + laneId;
            uint gj = TILE_N * tileID.x + lN * j;
            outputO.x[coordToOffset(gi, gj, strideC)/4] = C0[i][j];
            outputO.x[coordToOffset(gi, gj, strideC)/4+1] = C1[i][j];
            outputO.x[coordToOffset(gi, gj, strideC)/4+2] = C2[i][j];
            outputO.x[coordToOffset(gi, gj, strideC)/4+3] = C3[i][j];
        }
    }
}