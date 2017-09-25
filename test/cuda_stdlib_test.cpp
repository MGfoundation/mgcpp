#include <gtest/gtest.h>

#include <mgcpp/cuda/stdlib.hpp>
#include <mgcpp/cuda/internal/status_wrapper.hpp>

TEST(cuda_malloc, cuda_malloc_success)
{
    using mgcpp::internal::cuda_mem_get_info;

    float* ptr = nullptr;

    size_t free_memory_before_malloc = 0;
    cuda_mem_get_info(&free_memory_before_malloc, nullptr);

    EXPECT_NO_THROW({ptr = mgcpp::cuda_malloc<float>(10);});
    EXPECT_NE(ptr, nullptr);

    size_t free_memory_after_malloc = 0;
    cuda_mem_get_info(&free_memory_after_malloc, nullptr);

    EXPECT_GT(free_memory_before_malloc, free_memory_after_malloc);

    mgcpp::cuda_free(ptr);
}

TEST(cuda_malloc, cuda_malloc_throw_failure)
{
    using mgcpp::internal::cuda_mem_get_info;

    float* ptr = nullptr;

    size_t free_memory= 0;
    cuda_mem_get_info(&free_memory, nullptr);

    EXPECT_ANY_THROW(
        {ptr = mgcpp::cuda_malloc<float>(free_memory * 2);});
}

TEST(cuda_malloc, cuda_malloc_nothrow_failure)
{
    using mgcpp::internal::cuda_mem_get_info;

    float* ptr = nullptr;

    size_t free_memory= 0;
    cuda_mem_get_info(&free_memory, nullptr);

    EXPECT_NO_THROW([&](){
            ptr = mgcpp::cuda_malloc<float>(free_memory * 2,
                                            std::nothrow);
        }());

    EXPECT_EQ(ptr, nullptr);
}

TEST(cuda_malloc, cuda_malloc_nothrow_success)
{
    using mgcpp::internal::cuda_mem_get_info;

    size_t free_memory_before_malloc = 0;
    cuda_mem_get_info(&free_memory_before_malloc, nullptr);

    float* ptr = nullptr;
    ptr = mgcpp::cuda_malloc<float>(10, std::nothrow);
    EXPECT_NE(ptr, nullptr);

    size_t free_memory_after_malloc = 0;
    cuda_mem_get_info(&free_memory_after_malloc, nullptr);

    EXPECT_GT(free_memory_before_malloc, free_memory_after_malloc);

    mgcpp::cuda_free(ptr);
}

TEST(cuda_free, cuda_free_success)
{
    using mgcpp::internal::cuda_mem_get_info;

    size_t free_memory_before_malloc = 0;
    cuda_mem_get_info(&free_memory_before_malloc, nullptr);

    float* ptr = nullptr;

    EXPECT_NO_THROW(
        [&ptr](){ptr = mgcpp::cuda_malloc<float>(10);}());
    EXPECT_NE(ptr, nullptr);

    size_t free_memory_after_malloc = 0;
    cuda_mem_get_info(&free_memory_after_malloc, nullptr);

    EXPECT_GT(free_memory_before_malloc, free_memory_after_malloc);

    bool success = mgcpp::cuda_free(ptr);
    EXPECT_EQ(success, true);

    size_t free_memory_after_free = 0;
    cuda_mem_get_info(&free_memory_after_free, nullptr);

    EXPECT_EQ(free_memory_after_free, free_memory_before_malloc);
}

TEST(cuda_free, cuda_free_failure)
{
    float* ptr = (float*)10u;
    bool success = mgcpp::cuda_free(ptr);
    EXPECT_EQ(success, false);
}
