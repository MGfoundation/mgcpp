#include <mgcpp/cuda/internal/status_wrapper.hpp>
#include <mgcpp/cuda/internal/cuda_error.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>

TEST_CASE("cuda get memory status", "[cudaMemGetInfo]")
{
    using mgcpp::internal::cuda_error_t;
    using mgcpp::internal::cuda_mem_get_info;

    size_t free = 0;
    size_t total = 0;

    cudaError_t result = cudaMemGetInfo(&free, &total);

    REQUIRE(result == cudaError_t::cudaSuccess);

    size_t free_wrapper = 0;
    size_t total_wrapper = 0;

    cuda_error_t result_wrapper =
        cuda_mem_get_info(&free_wrapper, &total_wrapper);

    REQUIRE(
        static_cast<cuda_error_t>(result) == cuda_error_t::success);

    REQUIRE(free_wrapper == free);
    REQUIRE(total_wrapper == total);
}
