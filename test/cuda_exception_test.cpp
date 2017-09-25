#include <gtest/gtest.h>

#include <mgcpp/cuda/internal/status_wrapper.hpp>
#include <mgcpp/cuda/stdlib.hpp>
#include <mgcpp/cuda/exception.hpp>

TEST(mgcpp_exception, mgcpp_error_check)
{
    using mgcpp::internal::cuda_mem_get_info;

    float* ptr = nullptr;

    size_t free_memory= 0;
    cuda_mem_get_info(&free_memory, nullptr);

    EXPECT_DEATH({
            mgcpp_error_check(
                ptr = mgcpp::cuda_malloc<float>(free_memory * 2));
        }, "death by mgcpp_error_check");
}
