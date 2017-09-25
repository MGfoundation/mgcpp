#include <catch.hpp>

#include <mgcpp/cuda/stdlib.hpp>
#include <mgcpp/cuda/internal/status_wrapper.hpp>

TEST_CASE("templated cuda malloc success", "[cuda_malloc]")
{
    using mgcpp::internal::cuda_mem_get_info;

    float* ptr = nullptr;

    size_t free_memory_before_malloc = 0;
    cuda_mem_get_info(&free_memory_before_malloc, nullptr);

    REQUIRE_NOTHROW([&ptr](){ptr = mgcpp::cuda_malloc<float>(10);}());
    REQUIRE(ptr != nullptr);

    size_t free_memory_after_malloc = 0;
    cuda_mem_get_info(&free_memory_after_malloc, nullptr);

    REQUIRE(free_memory_before_malloc > free_memory_after_malloc);

    mgcpp::cuda_free(ptr);
}

TEST_CASE("templated cuda malloc throws failure",
          "[cuda_malloc]")
{
    using mgcpp::internal::cuda_mem_get_info;

    float* ptr = nullptr;

    size_t free_memory= 0;
    cuda_mem_get_info(&free_memory, nullptr);

    REQUIRE_THROWS([&](){
            ptr = mgcpp::cuda_malloc<float>(free_memory * 2);}());
}

TEST_CASE("templated cuda malloc nothrow failure",
          "[cuda_malloc]")
{
    using mgcpp::internal::cuda_mem_get_info;

    float* ptr = nullptr;

    size_t free_memory= 0;
    cuda_mem_get_info(&free_memory, nullptr);

    REQUIRE_NOTHROW([&](){
            ptr = mgcpp::cuda_malloc<float>(free_memory * 2,
                                            std::nothrow);
        }());

    REQUIRE(ptr == nullptr);
}

TEST_CASE("templated cuda malloc nothrow success",
          "[cuda_malloc][cuda_free]")
{
    using mgcpp::internal::cuda_mem_get_info;

    size_t free_memory_before_malloc = 0;
    cuda_mem_get_info(&free_memory_before_malloc, nullptr);

    float* ptr = nullptr;
    ptr = mgcpp::cuda_malloc<float>(10, std::nothrow);
    REQUIRE(ptr != nullptr);

    size_t free_memory_after_malloc = 0;
    cuda_mem_get_info(&free_memory_after_malloc, nullptr);

    REQUIRE(free_memory_before_malloc > free_memory_after_malloc);

    mgcpp::cuda_free(ptr);
}

TEST_CASE("templated cuda malloc and free success",
          "[cuda_malloc][cuda_free]")
{
    using mgcpp::internal::cuda_mem_get_info;

    size_t free_memory_before_malloc = 0;
    cuda_mem_get_info(&free_memory_before_malloc, nullptr);

    float* ptr = nullptr;

    REQUIRE_NOTHROW(
        [&ptr](){ptr = mgcpp::cuda_malloc<float>(10);}());
    REQUIRE(ptr != nullptr);

    size_t free_memory_after_malloc = 0;
    cuda_mem_get_info(&free_memory_after_malloc, nullptr);

    REQUIRE(free_memory_before_malloc > free_memory_after_malloc);

    bool success = mgcpp::cuda_free(ptr);
    REQUIRE(success == true);

    size_t free_memory_after_free = 0;
    cuda_mem_get_info(&free_memory_after_free, nullptr);

    REQUIRE(free_memory_after_free == free_memory_before_malloc);
}


TEST_CASE("templated cuda malloc and free nothrow success",
          "[cuda_malloc]")
{
    using mgcpp::internal::cuda_mem_get_info;

    size_t free_memory_before_malloc = 0;
    cuda_mem_get_info(&free_memory_before_malloc, nullptr);

    float* ptr = nullptr;
    ptr = mgcpp::cuda_malloc<float>(10, std::nothrow);
    REQUIRE(ptr != nullptr);

    size_t free_memory_after_malloc = 0;
    cuda_mem_get_info(&free_memory_after_malloc, nullptr);

    REQUIRE(free_memory_before_malloc > free_memory_after_malloc);

    bool free_result = mgcpp::cuda_free(ptr);
    REQUIRE(free_result == true);

    size_t free_memory_after_free = 0;
    cuda_mem_get_info(&free_memory_after_free, nullptr);

    REQUIRE(free_memory_after_free == free_memory_before_malloc);
}

TEST_CASE("templated cuda free failure", "[cuda_free]")
{
    float* ptr = (float*)10u;
    bool success = mgcpp::cuda_free(ptr);
    REQUIRE(success == false);
}
