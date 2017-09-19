#include <catch.hpp>

#include <mgcpp/cuda/internal/cuda_stdlib.hpp>

using mgcpp::internal::cuda_error_t;

TEST_CASE("cuda malloc success", "[cudaMalloc]")
{
    float* ptr = nullptr;
    cuda_error_t result =
        mgcpp::internal::cuda_malloc((void*)ptr,
                                     sizeof(float) * 10);

    REQUIRE( result == cuda_error_t::success );
}

TEST_CASE("cuda free success", "[cudaFree]")
{
    float* ptr = nullptr;
    cuda_error_t malloc_result =
        mgcpp::internal::cuda_malloc((void*)ptr,
                                     sizeof(float) * 10);
    REQUIRE( malloc_result == cuda_error_t::success );

    cuda_error_t free_result = 
        mgcpp::internal::cuda_free((void*)ptr);
    REQUIRE( free_result == cuda_error_t::success );
}
