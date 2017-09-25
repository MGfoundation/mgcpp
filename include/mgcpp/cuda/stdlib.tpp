#include <mgcpp/cuda/internal/stdlib_wrapper.hpp>
#include <mgcpp/cuda/internal/cuda_error.hpp>
#include <mgcpp/cuda/stdlib.hpp>
#include <mgcpp/cuda/exception.hpp>

namespace mgcpp
{
    template<typename ElemType, typename>
    ElemType*
    cuda_malloc(size_t size)
    {
        using internal::cuda_malloc;
        using internal::cuda_error_t;

        void* ptr = nullptr;
        cuda_error_t err_code =
            cuda_malloc(&ptr, size * sizeof(ElemType));

        if(err_code != cuda_error_t::success)
            MGCPP_THROW_BAD_ALLOC;

        return static_cast<ElemType*>(ptr);
    }

    template<typename ElemType, typename>
    ElemType*
    cuda_malloc(size_t size,
                std::nothrow_t const& nothrow_flag) noexcept
    {
        (void)nothrow_flag; // warning suppression

        using internal::cuda_malloc;
        using internal::cuda_error_t;

        void* ptr = nullptr;
        cuda_error_t err_code =
            cuda_malloc(&ptr, size * sizeof(ElemType));

        if(err_code != cuda_error_t::success)
            return nullptr;

        return static_cast<ElemType*>(ptr);
    }

    template<typename ElemType>
    bool
    cuda_free(ElemType* ptr) noexcept
    {
        using internal::cuda_free;
        using internal::cuda_error_t;

        cuda_error_t err_code = cuda_free(ptr);

        if(err_code != cuda_error_t::success)
            return false;
        return true;
    }
}
