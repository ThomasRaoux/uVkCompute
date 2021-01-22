#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_AMD_gpu_shader_half_float: enable

layout(binding=0) buffer InputA { vec2 x[]; } inputA;
layout(binding=1) buffer InputB { vec4 x[]; } inputB;
layout(binding=2) buffer Output { uvec4 x[]; } outputO;

const uint sX = 8;
const uint sY = 2;

layout(local_size_x = sX, local_size_y = sY, local_size_z = 1) in;

layout(constant_id = 0) const uint M = 1;
layout(constant_id = 1) const uint N = 1;
layout(constant_id = 2) const uint K = 1;

const uint strideA = K;
const uint strideB = N;
const uint strideC = N;

const uint C_ROWS = TILE_M / sY;
const uint C_COLS = TILE_N / (4*sX);

layout(set = 0, binding = 3) uniform sampler2D texB;

uint coordToOffset(uint i, uint j, uint stride)
{
    return (stride * i + j);
}

void main()
{
    uvec2 gID = gl_WorkGroupID.xy;
    uvec2 laneId = gl_LocalInvocationID.xy;
    f16vec4 C[C_ROWS][C_COLS];
    f16vec4 B[TILE_K][C_COLS];

    // Initialize result to zero
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            C[i][j] = f16vec4(0.f, 0.f, 0.f, 0.f);
        }
    }

    for (uint k = 0; k < K; k+=TILE_K) {
       [[unroll]] for (uint j = 0; j < C_COLS; j+=2) {
          [[unroll]] for (uint i = 0; i < TILE_K; ++i) {
            uint gj = gID.x * TILE_N/4 + laneId.x*2 + j*sX;
            uint gk = k+i;
            vec4 temp;

          #if (TEXTURE == 1)
            temp = texelFetch(texB, ivec2(gj/2, gk), 0);
          #else
            uint offset = coordToOffset(gk, gj/2, strideB/8);
            offset = (gk < K && gj < N/4) ? offset : 0;
            temp = inputB.x[offset];
          #endif
          #if (UNALIGNED == 1)  
            temp = (gk < K && gj < N/4) ? temp : vec4(0.f, 0.f, 0.f, 0.f);
          #endif  
            B[i][j].x = unpackFloat2x16(floatBitsToUint(temp.x)).x;
            B[i][j].y = unpackFloat2x16(floatBitsToUint(temp.x)).y;
            B[i][j].z = unpackFloat2x16(floatBitsToUint(temp.y)).x;
            B[i][j].w = unpackFloat2x16(floatBitsToUint(temp.y)).y;
            B[i][j+1].x = unpackFloat2x16(floatBitsToUint(temp.z)).x;
            B[i][j+1].y = unpackFloat2x16(floatBitsToUint(temp.z)).y;
            B[i][j+1].z = unpackFloat2x16(floatBitsToUint(temp.w)).x;
            B[i][j+1].w = unpackFloat2x16(floatBitsToUint(temp.w)).y;
          }
        }

        [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
          uint gi = gID.y * TILE_M + laneId.y + i*sY;
          uint gk = k/4;
          [[unroll]] for (uint kk = 0; kk < TILE_K/4; kk++) {
          vec2 temp;
          uint offset = coordToOffset(gi, gk+kk, strideA/4);
        #if (UNALIGNED == 1)  
          offset = (gi < M && gk < K/4) ? offset : 0;
        #endif   
          temp = inputA.x[offset];
        #if (UNALIGNED == 1)  
          temp = (gi < M && gk < K/4) ? temp : vec2(0.f, 0.f);
        #endif      
            float16_t a;
            [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
              a = unpackFloat2x16(floatBitsToUint(temp.x)).x;
              C[i][j] += f16vec4(a, a, a, a)*B[0+4*kk][j];
              a = unpackFloat2x16(floatBitsToUint(temp.x)).y;
              C[i][j] += f16vec4(a, a, a, a)*B[1+4*kk][j];
              a = unpackFloat2x16(floatBitsToUint(temp.y)).x;
              C[i][j] += f16vec4(a, a, a, a)*B[2+4*kk][j];
              a = unpackFloat2x16(floatBitsToUint(temp.y)).y;
              C[i][j] += f16vec4(a, a, a, a)*B[3+4*kk][j];
            }
          }
        }
    }

    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; j+=2) {
            uint gi = gID.y * TILE_M + laneId.y + i*sY;
            uint gj = gID.x * TILE_N/4 + laneId.x*2 + j*sX;
            uvec4 temp;
            temp.x = packFloat2x16(f16vec2(C[i][j].xy));
            temp.y = packFloat2x16(f16vec2(C[i][j].zw));
            temp.z = packFloat2x16(f16vec2(C[i][j+1].xy));
            temp.w = packFloat2x16(f16vec2(C[i][j+1].zw));
          #if (UNALIGNED == 1)
            if(gi < M && gj < N/4) {
          #endif
              outputO.x[gi * strideC/8 + gj/2] = temp;
          #if (UNALIGNED == 1)              
            }
          #endif
        }
    }
}
