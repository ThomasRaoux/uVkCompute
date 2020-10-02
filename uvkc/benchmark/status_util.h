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

#ifndef UVKC_BENCHMARK_STATUS_UTIL_H_
#define UVKC_BENCHMARK_STATUS_UTIL_H_

#include "uvkc/base/log.h"
#include "uvkc/base/status.h"

namespace uvkc {
namespace benchmark {

//===----------------------------------------------------------------------===//
// Utility macros
//===----------------------------------------------------------------------===//

#define BM_CHECK(condition)            \
  (condition ? ::uvkc::GetNullLogger() \
             : ::uvkc::benchmark::CheckError(__FILE__, __LINE__).logger())

#define BM_CHECK_EQ(a, b) BM_CHECK((a) == (b))
#define BM_CHECK_NE(a, b) BM_CHECK((a) != (b))

#define BM_CHECK_OK(rexpr)                                               \
  BM_CHECK_OK_INNER_(__FILE__, __LINE__,                                 \
                     UVKC_STATUS_IMPL_CONCAT_(_status_object, __LINE__), \
                     rexpr)

#define BM_CHECK_OK_AND_ASSIGN(lhs, rexpr) \
  BM_CHECK_OK_AND_ASSIGN_INNER_(           \
      __FILE__, __LINE__,                  \
      UVKC_STATUS_IMPL_CONCAT_(_status_or_object, __LINE__), lhs, rexpr)

//===----------------------------------------------------------------------===//
// Utility class
//===----------------------------------------------------------------------===//

// A wrapper that prints 'file:line: check error: ' prefix and '\n' suffix for
// an error message and aborts the program.
class CheckError {
 public:
  CheckError(const char *file, int line);
  ~CheckError();

  Logger &logger() { return logger_; }

 private:
  CheckError(const CheckError &) = delete;
  CheckError &operator=(const CheckError &) = delete;

  Logger &logger_;
};

//===----------------------------------------------------------------------===//
// Macros internals
//===----------------------------------------------------------------------===//

#define BM_CHECK_OK_INNER_(file, line, status, rexpr)                      \
  do {                                                                     \
    auto status = rexpr;                                                   \
    if (!status.ok()) {                                                    \
      ::uvkc::GetErrorLogger()                                             \
          << file << ":" << line << ": check error: " << status.ToString() \
          << "\n";                                                         \
      std::abort();                                                        \
    }                                                                      \
  } while (0)

#define BM_CHECK_OK_AND_ASSIGN_INNER_(file, line, statusor, lhs, rexpr) \
  auto statusor = rexpr;                                                \
  if (!statusor.ok()) {                                                 \
    ::uvkc::GetErrorLogger()                                            \
        << file << ":" << line                                          \
        << ": check error: " << std::move(statusor).status().ToString() \
        << "\n";                                                        \
    std::abort();                                                       \
  }                                                                     \
  lhs = std::move(statusor).value()

}  // namespace benchmark
}  // namespace uvkc

#endif  // UVKC_BENCHMARK_STATUS_UTIL_H_
