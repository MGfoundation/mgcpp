#include <catch.hpp>

#include <mgcpp/cuda/internal/stdlib_wrapper.hpp>
#include <mgcpp/cuda/internal/status_wrapper.hpp>

using mgcpp::internal::cuda_error_t;

TEST_CASE("cuda malloc success", "[cuda_malloc]")
{
    using mgcpp::internal::cuda_mem_get_info;
    using mgcpp::internal::cuda_malloc;

    size_t free_memory_before = 0;
    cuda_mem_get_info(&free_memory_before, nullptr);

    float* ptr = nullptr;
    cuda_error_t result = cuda_malloc((void**)&ptr,
                                      sizeof(float) * 10);

    size_t free_memory_after = 0;
    cuda_mem_get_info(&free_memory_after, nullptr);

    REQUIRE(free_memory_before > free_memory_after);

    REQUIRE( result == cuda_error_t::success );
    mgcpp::internal::cuda_free(ptr);
}

TEST_CASE("cuda malloc and free success", "[cuda_malloc][cuda_free]")
{
    using mgcpp::internal::cuda_mem_get_info;
    using mgcpp::internal::cuda_malloc;

    size_t free_memory_before = 0;
    cuda_mem_get_info(&free_memory_before, nullptr);

    float* ptr = nullptr;
    cuda_error_t malloc_result =
        mgcpp::internal::cuda_malloc((void**)&ptr,
                                     sizeof(float) * 10);
    REQUIRE( malloc_result == cuda_error_t::success );

    size_t free_memory_after_malloc = 0;
    cuda_mem_get_info(&free_memory_after_malloc, nullptr);

    REQUIRE(free_memory_before > free_memory_after_malloc);

    cuda_error_t free_result = 
        mgcpp::internal::cuda_free((void*)ptr);
    REQUIRE( free_result == cuda_error_t::success );

    size_t free_memory_after_free = 0;
    cuda_mem_get_info(&free_memory_after_free, nullptr);

    REQUIRE(free_memory_after_free == free_memory_before);
}
