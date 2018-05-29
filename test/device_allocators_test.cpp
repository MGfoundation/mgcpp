
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/allocators/allocator.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cuda/memory.hpp>

TEST(default_allocator, device_allocation_and_deallocation) {
  auto stat = mgcpp::cuda_set_device(0);
  EXPECT_TRUE(bool(stat));

  auto mem_info = mgcpp::cuda_mem_get_info();
  size_t before_memory = mem_info.value().first;

  mgcpp::allocator<float> allocator{};

  float* ptr = nullptr;
  EXPECT_NO_THROW({ ptr = allocator.allocate_device(10); });
  EXPECT_NE(ptr, nullptr);

  mem_info = mgcpp::cuda_mem_get_info();
  size_t after_memory = mem_info.value().first;

  EXPECT_GT(before_memory, after_memory);

  EXPECT_NO_THROW({ allocator.deallocate_device(ptr, 10); });

  mem_info = mgcpp::cuda_mem_get_info();
  size_t final_memory = mem_info.value().first;

  EXPECT_EQ(before_memory, final_memory);
}

TEST(default_allocator, copy_to_and_from_host) {
  auto stat = mgcpp::cuda_set_device(0);
  EXPECT_TRUE(bool(stat));

  mgcpp::allocator<float> allocator{};

  size_t size = 10;

  float* host = allocator.allocate_host(size);
  float* device = allocator.allocate_device(size);

  *host = 10;

  EXPECT_NO_THROW({ allocator.copy_from_host(device, host, size); });
  EXPECT_NO_THROW({ allocator.copy_to_host(host, device, size); });

  EXPECT_EQ(*host, 10);

  allocator.deallocate_host(host, size);
  allocator.deallocate_device(device, size);
}
