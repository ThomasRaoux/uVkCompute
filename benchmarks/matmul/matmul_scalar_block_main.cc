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
#include "uvkc/benchmark/main.h"
#include "uvkc/benchmark/status_util.h"
#include "uvkc/benchmark/vulkan_buffer_util.h"
#include "uvkc/benchmark/vulkan_context.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/vulkan/pipeline.h"

static const char kBenchmarkName[] = "matmul_scalar_block";

#include "matmul_scalar_block_shader_spirv_permutation.inc"

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
    vulkan::Device *device, const LatencyMeasure *latency_measure) {}

}  // namespace benchmark
}  // namespace uvkc
