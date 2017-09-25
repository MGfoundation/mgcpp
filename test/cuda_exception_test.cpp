#include <catch.hpp>

#include <mgcpp/cuda/stdlib.hpp>
#include <mgcpp/cuda/exception.hpp>
#include <mgcpp/cuda/internal/status_wrapper.hpp>

TEST_CASE("templated cuda malloc throws failure",
          "[cuda_malloc]")
{
    using mgcpp::internal::cuda_mem_get_info;

    float* ptr = nullptr;

    size_t free_memory= 0;
    cuda_mem_get_info(&free_memory, nullptr);

    REQUIRE_NOTHROW(
        [&](){
            mgcpp_error_check(
                ptr = mgcpp::cuda_malloc<float>(free_memory * 2));
        }());
}
