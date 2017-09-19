#include <catch.hpp>

#include <limits>

#include <mgcpp/cuda/cuda_template_stdlib.hpp>

TEST_CASE("templated cuda malloc success", "[cuda_malloc]")
{
    float* ptr = nullptr;

    REQUIRE_NOTHROW([ptr](){float* ptr = cuda_malloc<float>(10)}());
    REQUIRE(ptr != nullptr);

    cuda_free(ptr);
}

TEST_CASE("templated cuda malloc success", "[cuda_malloc][cuda_free]")
{
    float* ptr = nullptr;

    REQUIRE_NOTHROW([ptr](){float* ptr = cuda_malloc<float>(10)}());
    REQUIRE(ptr != nullptr);

    cuda_free(ptr);

    REQUIRE_NOTHROW(cuda_free(ptr););
}

TEST_CASE("templated cuda malloc and free nothrow success", "[cuda_malloc]")
{
    float* ptr = nullptr;
    ptr = cuda_malloc_nothrow(10);
    REQUIRE(ptr != nullptr);

    bool free_result = cuda_free_nothrow(ptr);
    REQUIRE(free_result == true);
}

TEST_CASE("templated cuda malloc nothrow success", "[cuda_malloc][cuda_free]")
{
    float* ptr = nullptr;
    ptr = cuda_malloc_nothrow(10);
    REQUIRE(ptr != nullptr);

    cuda_free(ptr);
}

TEST_CASE("templated cuda malloc throws failure", "[cuda_malloc][cuda_free]")
{
    float* ptr = nullptr;

    REQUIRE_THROWS(
        [ptr](){float* ptr = cuda_malloc<float>(
                std::numeric_limits<size_t>::max())}());
    REQUIRE(ptr != nullptr);

    cuda_free(ptr);

    REQUIRE_NOTHROW(cuda_free(ptr););
}

TEST_CASE("templated cuda free throws failure", "[cuda_free]")
{
    float* ptr = nullptr;
    REQUIRE_THROWS(cuda_free(ptr));
}

TEST_CASE("templated cuda free nothrow failure", "[cuda_free]")
{
    float* ptr = nullptr;
    bool result = cuda_free_nothrow(ptr);
    REQUIRE(result == false);
}
