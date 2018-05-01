
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/cuda/memory.hpp>

TEST(cuda_malloc, cuda_malloc_success) {
  size_t free_memory_before_malloc = 0;
  cudaMemGetInfo(&free_memory_before_malloc, nullptr);

  auto rst = mgcpp::cuda_malloc<float>(10);
  EXPECT_TRUE(bool(rst));

  size_t free_memory_after_malloc = 0;
  cudaMemGetInfo(&free_memory_after_malloc, nullptr);

  EXPECT_GT(free_memory_before_malloc, free_memory_after_malloc);

  (void)mgcpp::cuda_free(rst.value());
}

TEST(cuda_malloc, cuda_malloc_failure) {
  size_t free_memory = 0;
  cudaMemGetInfo(&free_memory, nullptr);

  auto ptr = mgcpp::cuda_malloc<float>(free_memory * 2);
  EXPECT_FALSE(ptr);
}

TEST(cuda_free, cuda_free_success) {
  size_t free_memory_before_malloc = 0;
  cudaMemGetInfo(&free_memory_before_malloc, nullptr);

  auto result = mgcpp::cuda_malloc<float>(10);
  ;
  EXPECT_TRUE(bool(result));

  size_t free_memory_after_malloc = 0;
  cudaMemGetInfo(&free_memory_after_malloc, nullptr);

  EXPECT_GT(free_memory_before_malloc, free_memory_after_malloc);

  auto free_result = mgcpp::cuda_free(result.value());
  EXPECT_TRUE(bool(free_result));

  size_t free_memory_after_free = 0;
  cudaMemGetInfo(&free_memory_after_free, nullptr);

  EXPECT_EQ(free_memory_after_free, free_memory_before_malloc);
}

TEST(cuda_free, cuda_free_failure) {
  float* ptr = (float*)10u;
  auto result = mgcpp::cuda_free(ptr);
  EXPECT_FALSE(result);
}

TEST(cuda_memcpy, memcpy_to_and_from_host) {
  size_t size = 1;
  auto device = mgcpp::cuda_malloc<float>(size);
  float host = 7;

  host = 7;
  auto to_device_stat = mgcpp::cuda_memcpy(
      device.value(), &host, size, mgcpp::cuda_memcpy_kind::host_to_device);
  EXPECT_TRUE(bool(to_device_stat));

  host = 0;
  auto to_host_stat = mgcpp::cuda_memcpy(
      &host, device.value(), size, mgcpp::cuda_memcpy_kind::device_to_host);
  EXPECT_TRUE(bool(to_host_stat));

  EXPECT_EQ(host, 7);
  (void)mgcpp::cuda_free(device.value());
}

TEST(cuda_memset, memset_to_zero) {
  size_t size = 1;
  auto memory = mgcpp::cuda_malloc<float>(size);

  float host = 7;

  auto to_device_stat = mgcpp::cuda_memcpy(
      memory.value(), &host, size, mgcpp::cuda_memcpy_kind::host_to_device);
  EXPECT_TRUE(bool(to_device_stat));

  auto status = mgcpp::cuda_memset(memory.value(), 0.0f, size);
  EXPECT_TRUE(bool(status));

  host = 7;
  auto to_host_stat = mgcpp::cuda_memcpy(
      &host, memory.value(), size, mgcpp::cuda_memcpy_kind::device_to_host);
  EXPECT_TRUE(bool(to_host_stat));
  EXPECT_EQ(host, 0);

  (void)mgcpp::cuda_free(memory.value());
}
