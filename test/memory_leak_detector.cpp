
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/exception.hpp>

#include <string>

#include "memory_leak_detector.hpp"
#include "test_policy.hpp"

namespace mgcpp {
void memory_leak_detector::OnTestStart(::testing::TestInfo const& test_info) {
  (void)cuda_set_device(0);

  if (!test_policy::get_policy().detect_memory_leak())
    return;

  if (test_info.name() == std::string("thread_context"))
    return;

  auto device_number = test_policy::get_policy().device_num();

  device_free_memory.clear();
  device_free_memory.reserve(device_number);

  for (auto i = 0u; i < device_number; ++i) {
    (void)cuda_set_device(i);

    auto memstat = cuda_mem_get_info();
    EXPECT_TRUE(bool(memstat))
        << "error occurred while getting memory of device " << i << '\n'
        << memstat.error() << "in start of test " << test_info.name();

    device_free_memory.push_back(memstat.value().first);
  }
}

void memory_leak_detector::OnTestEnd(::testing::TestInfo const& test_info) {
  (void)cuda_set_device(0);

  if (!test_policy::get_policy().detect_memory_leak())
    return;

  if (test_info.test_case_name() == std::string("thread_context"))
    return;

  auto device_number = test_policy::get_policy().device_num();
  for (auto i = 0u; i < device_number; ++i) {
    (void)cuda_set_device(i);

    auto memstat = cuda_mem_get_info();

    EXPECT_TRUE(bool(memstat))
        << "error while getting memory info of device " << i << '\n'
        << "in test " << test_info.name();

    size_t result_memory = memstat.value().first;
    EXPECT_EQ(result_memory, device_free_memory[i])
        << "memory leak detected in test " << test_info.name() << '\n'
        << "for device " << i << '\n';
  }
}
}  // namespace mgcpp
