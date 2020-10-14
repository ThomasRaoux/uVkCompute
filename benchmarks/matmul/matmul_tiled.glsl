#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_EXT_control_flow_attributes : enable

const uint subgroupsize = 16;

layout(binding=0) buffer InputA { vec4 x[]; } inputA;
layout(binding=1) buffer InputB { float x[]; } inputB;
layout(binding=2) buffer Output { float x[]; } outputO;
layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint M = 1;
layout(constant_id = 1) const uint N = 1;
layout(constant_id = 2) const uint K = 1;

const uint strideA = K;
const uint strideB = N;
const uint strideC = N;

const uint lM = 4;
const uint lN = subgroupsize;

const uint C_ROWS = TILE_M / 4;
const uint C_COLS = TILE_N / subgroupsize;

uint coordToOffset(uint i, uint j, uint stride)
{
    return (stride * i + j);
}

void matmul4xSxS(inout vec4 C, vec4 A, float B0, float B1, float B2, float B3, uint k, uint j, uint i) {
    C.x += B0 * subgroupShuffle(A.x, i);
    C.y += B0 * subgroupShuffle(A.x, i+subgroupsize/4);
    C.z += B0 * subgroupShuffle(A.x, i+2*subgroupsize/4);
    C.w += B0 * subgroupShuffle(A.x, i+3*(subgroupsize/4));

    C.x += B1 * subgroupShuffle(A.y, i);
    C.y += B1 * subgroupShuffle(A.y, i+subgroupsize/4);
    C.z += B1 * subgroupShuffle(A.y, i+2*subgroupsize/4);
    C.w += B1 * subgroupShuffle(A.y, i+3*(subgroupsize/4));

    C.x += B2 * subgroupShuffle(A.z, i);
    C.y += B2 * subgroupShuffle(A.z, i+subgroupsize/4);
    C.z += B2 * subgroupShuffle(A.z, i+2*subgroupsize/4);
    C.w += B2 * subgroupShuffle(A.z, i+3*(subgroupsize/4));

    C.x += B3 * subgroupShuffle(A.w, i);
    C.y += B3 * subgroupShuffle(A.w, i+subgroupsize/4);
    C.z += B3 * subgroupShuffle(A.w, i+2*subgroupsize/4);
    C.w += B3 * subgroupShuffle(A.w, i+3*(subgroupsize/4));
}

void main()
{
    vec4 C[C_ROWS][C_COLS];
    uvec2 tileID = uvec2(gl_WorkGroupID.xy);

    // Initialize result to zero
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            C[i][j] = vec4(0.f, 0.f, 0.f, 0.f);
        }
    }
    uint laneId = gl_LocalInvocationID.x;
    for (uint chunkK = 0; chunkK < K; chunkK += subgroupsize) {
        vec4 matA[C_ROWS];
        [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
            uint gi = TILE_M * tileID.y + lM * i +(laneId*4) /subgroupsize;
            uint gk = chunkK + (laneId*4) % subgroupsize;
            matA[i] = inputA.x[coordToOffset(gi, gk, strideA)/4];
        }
        [[unroll]] for (uint k = 0; k < subgroupsize/4; ++k) {

        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gj = TILE_N * tileID.x + lN * j + laneId;
            uint gk = chunkK;
            float B0 = inputB.x[coordToOffset(gk+k*4, gj, strideB)];
            float B1 = inputB.x[coordToOffset(gk+k*4+1, gj, strideB)];
            float B2 = inputB.x[coordToOffset(gk+k*4+2, gj, strideB)];
            float B3 = inputB.x[coordToOffset(gk+k*4+3, gj, strideB)];
            [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
              matmul4xSxS(C[i][j], matA[i], B0, B1, B2, B3, gk, gj, k);
            }
        }
        }
    }
        
    // Initialize result to zero
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gi = TILE_M * tileID.y + lM * i;
            uint gj = TILE_N * tileID.x + lN * j + laneId;
            outputO.x[coordToOffset(gi, gj, strideC)] = C[i][j].x;
            outputO.x[coordToOffset(gi+1, gj, strideC)] = C[i][j].y;
            outputO.x[coordToOffset(gi+2, gj, strideC)] = C[i][j].z;
            outputO.x[coordToOffset(gi+3, gj, strideC)] = C[i][j].w;
        }
    }
}