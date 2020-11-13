// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <memory>
#include <numeric>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "uvkc/benchmark/fp16_util.h"
#include "uvkc/benchmark/main.h"
#include "uvkc/benchmark/status_util.h"
#include "uvkc/benchmark/vulkan_buffer_util.h"
#include "uvkc/benchmark/vulkan_context.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/vulkan/pipeline.h"

using ::uvkc::benchmark::LatencyMeasureMode;

using namespace uvkc::benchmark;

static const char kBenchmarkName[] = "matmul_tiled";

#include "matmul_tiled_shader_spirv_permutation.inc"

struct ShaderCode {
  const char *name;       // Test case name
  const uint32_t *code;   // SPIR-V code
  size_t code_num_bytes;  // Number of bytes for SPIR-V code
  int tileM;
  int tileN;
  Precision precision;
};

#define SHADER_TILE_F32(A, B, C)                             \
  {#A "x" #B "x" #C "xf32",                                  \
   TILE_M_##A##_TILE_N_##B##_TILE_K_##C##_TYPE_vec4,         \
   sizeof(TILE_M_##A##_TILE_N_##B##_TILE_K_##C##_TYPE_vec4), \
   A,                                                        \
   B,                                                        \
   Precision::fp32},

#define SHADER_TILE_F16(A, B, C)                                \
  {#A "x" #B "x" #C "xf16",                                     \
   TILE_M_##A##_TILE_N_##B##_TILE_K_##C##_TYPE_f16vec4,         \
   sizeof(TILE_M_##A##_TILE_N_##B##_TILE_K_##C##_TYPE_f16vec4), \
   A,                                                           \
   B,                                                           \
   Precision::fp16},

#define SHADER_TILE(A, B, C) SHADER_TILE_F32(A, B, C) SHADER_TILE_F16(A, B, C)

// clang-format off
static ShaderCode kShaderCodeCases[] = {
  SHADER_TILE(1, 64,  4)
  SHADER_TILE(2, 64, 4)
  SHADER_TILE(3, 64, 4)
  SHADER_TILE(4, 64, 4)
  SHADER_TILE(5, 64, 4)
  SHADER_TILE(6, 64, 4)
  SHADER_TILE(7, 64, 4)
  SHADER_TILE(8, 64, 4)
  SHADER_TILE(9, 64, 4)
  SHADER_TILE(10, 64, 4)
  SHADER_TILE(11, 64, 4)
  SHADER_TILE(12, 64, 4)
  SHADER_TILE(13, 64, 4)
  SHADER_TILE(14, 64, 4)
  SHADER_TILE(15, 64, 4)
  SHADER_TILE(16, 64, 4)
  SHADER_TILE(1, 128, 4)
  SHADER_TILE(2, 128, 4)
  SHADER_TILE(3, 128, 4)
  SHADER_TILE(4, 128, 4)
  SHADER_TILE(5, 128, 4)
  SHADER_TILE(6, 128, 4)
  SHADER_TILE(7, 128, 4)
  SHADER_TILE(8, 128, 4)
  SHADER_TILE(9, 128, 4)
  SHADER_TILE(10, 128, 4)
  SHADER_TILE(11, 128, 4)
  SHADER_TILE(12, 128, 4)
  SHADER_TILE(13, 128, 4)
  SHADER_TILE(14, 128, 4)
  SHADER_TILE(15, 128, 4)
  SHADER_TILE(16, 128, 4)

  SHADER_TILE_F16(1, 64, 8)
  SHADER_TILE_F16(2, 64, 8)
  SHADER_TILE_F16(3, 64, 8)
  SHADER_TILE_F16(4, 64, 8)
  SHADER_TILE_F16(5, 64, 8)
  SHADER_TILE_F16(6, 64, 8)
  SHADER_TILE_F16(7, 64, 8)
  SHADER_TILE_F16(8, 64, 8)
  SHADER_TILE_F16(9, 64, 8)
  SHADER_TILE_F16(10, 64, 8)
  SHADER_TILE_F16(11, 64, 8)
  SHADER_TILE_F16(12, 64, 8)
  SHADER_TILE_F16(13, 64, 8)
  SHADER_TILE_F16(14, 64, 8)
  SHADER_TILE_F16(15, 64, 8)
  SHADER_TILE_F16(16, 64, 8)
  SHADER_TILE_F16(1, 128, 8)
  SHADER_TILE_F16(2, 128, 8)
  SHADER_TILE_F16(3, 128, 8)
  SHADER_TILE_F16(4, 128, 8)
  SHADER_TILE_F16(5, 128, 8)
  SHADER_TILE_F16(6, 128, 8)
  SHADER_TILE_F16(7, 128, 8)
  SHADER_TILE_F16(8, 128, 8)
  SHADER_TILE_F16(9, 128, 8)
  SHADER_TILE_F16(10, 128, 8)
  SHADER_TILE_F16(11, 128, 8)
  SHADER_TILE_F16(12, 128, 8)
  SHADER_TILE_F16(13, 128, 8)
  SHADER_TILE_F16(14, 128, 8)
  SHADER_TILE_F16(15, 128, 8)
  SHADER_TILE_F16(16, 128, 8)
};
// clang-format on

static void MatMul(::benchmark::State &state, ::uvkc::vulkan::Device *device,
                   const ::uvkc::benchmark::LatencyMeasure *latency_measure,
                   const uint32_t *code, size_t code_num_words, int M, int N,
                   int K, int tileM, int tileN, Precision precision) {
  //===-------------------------------------------------------------------===/
  // Create shader module, pipeline, and descriptor sets
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK_AND_ASSIGN(auto shader_module,
                         device->CreateShaderModule(code, code_num_words));

  ::uvkc::vulkan::Pipeline::SpecConstant spec_constant[3] = {};
  spec_constant[0].id = 0;
  spec_constant[0].type = ::uvkc::vulkan::Pipeline::SpecConstant::Type::s32;
  spec_constant[0].value.s32 = M;
  spec_constant[1].id = 1;
  spec_constant[1].type = ::uvkc::vulkan::Pipeline::SpecConstant::Type::s32;
  spec_constant[1].value.s32 = N;
  spec_constant[2].id = 2;
  spec_constant[2].type = ::uvkc::vulkan::Pipeline::SpecConstant::Type::s32;
  spec_constant[2].value.s32 = K;
  BM_CHECK_OK_AND_ASSIGN(
      auto pipeline, device->CreatePipeline(*shader_module, "main",
                                            absl::MakeSpan(spec_constant, 3)));

  BM_CHECK_OK_AND_ASSIGN(auto descriptor_pool,
                         device->CreateDescriptorPool(*shader_module));
  BM_CHECK_OK_AND_ASSIGN(auto layout_set_map,
                         descriptor_pool->AllocateDescriptorSets(
                             shader_module->descriptor_set_layouts()));

  //===-------------------------------------------------------------------===/
  // Create buffers
  //===-------------------------------------------------------------------===/
  const size_t src0_size = M * K * GetSize(precision);
  const size_t src1_size = K * N * GetSize(precision);
  const size_t dst_size = M * N * GetSize(precision);

  BM_CHECK_OK_AND_ASSIGN(
      auto src0_buffer,
      device->CreateBuffer(
          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, src0_size));
  BM_CHECK_OK_AND_ASSIGN(
      auto src1_buffer,
      device->CreateBuffer(
          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, src1_size));
  BM_CHECK_OK_AND_ASSIGN(
      auto dst_buffer,
      device->CreateBuffer(
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dst_size));

  //===-------------------------------------------------------------------===/
  // Set source buffer data
  //===-------------------------------------------------------------------===/
  auto getSrc0 = [K](int i, int j) {
    float v = ((float)((i + j * K) % 5) - 1.0f) / 2.0f;
    return v;
  };
  auto getSrc1 = [N](int i, int j) {
    float v = ((float)((i + j * N) % 7) - 1.0f) / 2.0f;
    return v;
  };

  if (precision == Precision::fp16) {
    BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
        device, src0_buffer.get(), src0_size, [&](void *ptr, size_t num_bytes) {
          uint16_t *src_float_buffer = reinterpret_cast<uint16_t *>(ptr);
          for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
              src_float_buffer[j + i * K] = fp16(getSrc0(i, j)).getValue();
            }
          }
        }));

    BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
        device, src1_buffer.get(), src1_size, [&](void *ptr, size_t num_bytes) {
          uint16_t *src_float_buffer = reinterpret_cast<uint16_t *>(ptr);
          for (int i = 0; i < K; i++) {
            for (int j = 0; j < N; j++) {
              src_float_buffer[j + i * N] = fp16(getSrc1(i, j)).getValue();
            }
          }
        }));
  } else if (precision == Precision::fp32) {
    BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
        device, src0_buffer.get(), src0_size, [&](void *ptr, size_t num_bytes) {
          float *src_float_buffer = reinterpret_cast<float *>(ptr);
          for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
              src_float_buffer[j + i * K] = getSrc0(i, j);
            }
          }
        }));

    BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
        device, src1_buffer.get(), src1_size, [&](void *ptr, size_t num_bytes) {
          float *src_float_buffer = reinterpret_cast<float *>(ptr);
          for (int i = 0; i < K; i++) {
            for (int j = 0; j < N; j++) {
              src_float_buffer[j + i * N] = getSrc1(i, j);
            }
          }
        }));
  }

  //===-------------------------------------------------------------------===/
  // Dispatch
  //===-------------------------------------------------------------------===/

  std::vector<::uvkc::vulkan::Device::BoundBuffer> bound_buffers(3);
  bound_buffers[0].buffer = src0_buffer.get();
  bound_buffers[0].set = 0;
  bound_buffers[0].binding = 0;
  bound_buffers[1].buffer = src1_buffer.get();
  bound_buffers[1].set = 0;
  bound_buffers[1].binding = 1;
  bound_buffers[2].buffer = dst_buffer.get();
  bound_buffers[2].set = 0;
  bound_buffers[2].binding = 2;
  BM_CHECK_OK(device->AttachBufferToDescriptor(
      *shader_module, layout_set_map,
      {bound_buffers.data(), bound_buffers.size()}));

  BM_CHECK_EQ(shader_module->descriptor_set_layouts().size(), 1)
      << "unexpected number of descriptor sets";
  auto descriptor_set_layout = shader_module->descriptor_set_layouts().front();

  std::vector<::uvkc::vulkan::CommandBuffer::BoundDescriptorSet>
      bound_descriptor_sets(1);
  bound_descriptor_sets[0].index = 0;
  bound_descriptor_sets[0].set = layout_set_map.at(descriptor_set_layout);
  BM_CHECK_OK_AND_ASSIGN(auto dispatch_cmdbuf, device->AllocateCommandBuffer());

  BM_CHECK_OK(dispatch_cmdbuf->Begin());
  dispatch_cmdbuf->BindPipelineAndDescriptorSets(
      *pipeline, {bound_descriptor_sets.data(), bound_descriptor_sets.size()});
  dispatch_cmdbuf->Dispatch(N / tileN, M / tileM, 1);
  BM_CHECK_OK(dispatch_cmdbuf->End());
  BM_CHECK_OK(device->QueueSubmitAndWait(*dispatch_cmdbuf));

  //===-------------------------------------------------------------------===/
  // Verify destination buffer data
  //===-------------------------------------------------------------------===/

  if (precision == Precision::fp16) {
    BM_CHECK_OK(::uvkc::benchmark::GetDeviceBufferViaStagingBuffer(
        device, dst_buffer.get(), dst_size, [&](void *ptr, size_t num_bytes) {
          uint16_t *dst_float_buffer = reinterpret_cast<uint16_t *>(ptr);
          for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
              float acc = 0.f;
              for (int k = 0; k < K; k++) {
                acc += getSrc0(i, k) * getSrc1(k, j);
              }
              float gpuValue = fp16(dst_float_buffer[j + i * N]).toFloat();
              BM_CHECK_FLOAT_EQ(gpuValue, acc, 0.5f)
                  << "destination buffer element (" << i << "," << j << ")"
                  << " has incorrect value: expected to be " << acc
                  << " but found " << gpuValue;
            }
          }
        }));
  } else if (precision == Precision::fp32) {
    BM_CHECK_OK(::uvkc::benchmark::GetDeviceBufferViaStagingBuffer(
        device, dst_buffer.get(), dst_size, [&](void *ptr, size_t num_bytes) {
          float *dst_float_buffer = reinterpret_cast<float *>(ptr);
          for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
              float acc = 0.f;
              for (int k = 0; k < K; k++) {
                acc += getSrc0(i, k) * getSrc1(k, j);
              }
              float gpuValue = dst_float_buffer[j + i * N];
              BM_CHECK_EQ(gpuValue, acc)
                  << "destination buffer element (" << i << "," << j << ")"
                  << " has incorrect value: expected to be " << acc
                  << " but found " << gpuValue;
            }
          }
        }));
  }

  //===-------------------------------------------------------------------===/
  // Benchmarking
  //===-------------------------------------------------------------------===/

  std::unique_ptr<::uvkc::vulkan::TimestampQueryPool> query_pool;
  bool use_timestamp =
      latency_measure->mode == LatencyMeasureMode::kGpuTimestamp;
  if (use_timestamp) {
    BM_CHECK_OK_AND_ASSIGN(query_pool, device->CreateTimestampQueryPool(2));
  }

  BM_CHECK_OK_AND_ASSIGN(auto cmdbuf, device->AllocateCommandBuffer());
  for (auto _ : state) {
    BM_CHECK_OK(cmdbuf->Begin());
    if (use_timestamp) cmdbuf->ResetQueryPool(*query_pool);

    cmdbuf->BindPipelineAndDescriptorSets(
        *pipeline,
        {bound_descriptor_sets.data(), bound_descriptor_sets.size()});

    if (use_timestamp) {
      cmdbuf->WriteTimestamp(*query_pool, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0);
    }

    cmdbuf->Dispatch(N / tileN, M / tileM, 1);

    if (use_timestamp) {
      cmdbuf->WriteTimestamp(*query_pool, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             1);
    }

    BM_CHECK_OK(cmdbuf->End());

    auto start_time = std::chrono::high_resolution_clock::now();
    BM_CHECK_OK(device->QueueSubmitAndWait(*cmdbuf));
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);

    switch (latency_measure->mode) {
      case LatencyMeasureMode::kSystemDispatch: {
        state.SetIterationTime(elapsed_seconds.count() -
                               latency_measure->overhead_seconds);
      } break;
      case LatencyMeasureMode::kSystemSubmit: {
        state.SetIterationTime(elapsed_seconds.count());
      } break;
      case LatencyMeasureMode::kGpuTimestamp: {
        BM_CHECK_OK_AND_ASSIGN(
            double timestamp_seconds,
            query_pool->CalculateElapsedSecondsBetween(0, 1));
        state.SetIterationTime(timestamp_seconds);
      } break;
    }

    BM_CHECK_OK(cmdbuf->Reset());
  }

  double numOperation = double(N) * double(M) * double(K) * 2.;
  state.counters["FLOps"] =
      ::benchmark::Counter(numOperation,
                           ::benchmark::Counter::kIsIterationInvariant |
                               ::benchmark::Counter::kIsRate,
                           ::benchmark::Counter::kIs1000);

  // Reset the command pool to release all command buffers in the benchmarking
  // loop to avoid draining GPU resources.
  BM_CHECK_OK(device->ResetCommandPool());
}

namespace uvkc {
namespace benchmark {

absl::StatusOr<std::unique_ptr<VulkanContext>> CreateVulkanContext() {
  return CreateDefaultVulkanContext(kBenchmarkName);
}

bool RegisterVulkanOverheadBenchmark(
    const vulkan::Driver::PhysicalDeviceInfo &physical_device,
    vulkan::Device *device, double *overhead_seconds) {
  return false;
}

void RegisterVulkanBenchmarks(
    const vulkan::Driver::PhysicalDeviceInfo &physical_device,
    vulkan::Device *device, const LatencyMeasure *latency_measure) {
  const char *gpu_name = physical_device.v10_properties.deviceName;

  const int M = 1024;
  const int N = 1024;
  const int K = 1024;
  for (Precision precision : {Precision::fp32, Precision::fp16}) {
    for (const auto &shader : kShaderCodeCases) {
      if (shader.precision != precision) continue;
      int paddM = (M + shader.tileM - 1) / shader.tileM * shader.tileM;
      int paddN = (N + shader.tileN - 1) / shader.tileN * shader.tileN;
      std::string test_name =
          absl::StrCat(gpu_name, "/", shader.name, "/", M, "x", N, "x", K, "/",
                       shader.tileM, "/", shader.tileN);
      ::benchmark::RegisterBenchmark(
          test_name.c_str(), MatMul, device, latency_measure, shader.code,
          shader.code_num_bytes / sizeof(uint32_t), paddM, paddN, K,
          shader.tileM, shader.tileN, shader.precision)
          ->UseManualTime()
          ->Unit(::benchmark::kMicrosecond);
    }
  }
}

}  // namespace benchmark
}  // namespace uvkc
