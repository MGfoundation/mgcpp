#include <string>

#include <catch.hpp>

#include <mgcpp/cuda/internal/cuda_error.hpp>

TEST_CASE("cuda error message", "[cudaGetErrorString]")
{
    using mgcpp::internal::cuda_error_t;
    using mgcpp::internal::cuda_error_string;

    cuda_error_t err_code = cuda_error_t::memory_allocation;
    auto result = std::string(cuda_error_string(err_code));
    auto answer = std::string("out of memory");

    REQUIRE(result == answer);
}




