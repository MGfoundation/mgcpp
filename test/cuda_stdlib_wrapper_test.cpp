#include <catch.hpp>

#include <mgcpp/cuda/internal/stdlib_wrapper.hpp>

using mgcpp::internal::cuda_error_t;

TEST_CASE("cuda malloc success", "[cuda_malloc]")
{
    float* ptr = nullptr;
    cuda_error_t result =
        mgcpp::internal::cuda_malloc((void**)&ptr,
                                     sizeof(float) * 10);

    REQUIRE( result == cuda_error_t::success );
    mgcpp::internal::cuda_free(ptr);
}

TEST_CASE("cuda malloc and free success", "[cuda_malloc][cuda_free]")
{
    float* ptr = nullptr;
    cuda_error_t malloc_result =
        mgcpp::internal::cuda_malloc((void**)&ptr,
                                     sizeof(float) * 10);
    REQUIRE( malloc_result == cuda_error_t::success );

    cuda_error_t free_result = 
        mgcpp::internal::cuda_free((void*)ptr);
    REQUIRE( free_result == cuda_error_t::success );
}
