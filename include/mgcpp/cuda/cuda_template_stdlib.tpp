#include <mgcpp/cuda/internal/cuda_stdlib.hpp>
#include <mgcpp/cuda/internal/cuda_error.hpp>
#include <mgcpp/cuda/cuda_template_stdlib.hpp>
#include <mgcpp/cuda/cuda_exceptions.hpp>

namespace mgcpp
{
    template<typename ElemType,
             typename std::enable_if<
                 std::is_arithmetic<ElemType>::value>>
    ElemType*
    cuda_malloc(size_t size)
    {
        using internal::cuda_malloc;
        using internal::cuda_error_t;

        void* ptr = nullptr;
        cuda_error_t err_code = cuda_malloc(ptr, size * sizeof(ElemType));

        if(err_code != cuda_error_t::cuda_success)
            throw cuda_bad_alloc(err_code);

        return static_cast<ElemType*>(ptr);
    }

    template<typename ElemType,
             typename std::enable_if<
                 std::is_arithmetic<ElemType>::value>>
    ElemType*
    cuda_malloc_nothrow(size_t size) noexcept
    {
        using internal::cuda_malloc;
        using internal::cuda_error_t;

        void* ptr = nullptr;
        cuda_error_t err_code = cuda_malloc(ptr, size * sizeof(ElemType));

        if(err_code != cuda_error_t::success)
            return nullptr;

        return static_cast<ElemType*>(ptr);
    }

    template<typename ElemType>
    void
    cuda_free(ElemType ptr)
    {
        using internal::cuda_free;
        using internal::cuda_error_t;

        cuda_error_t err_code = cuda_free(ptr);

        if(err_code != cuda_error_t::cuda_success)
            throw cuda_bad_dealloc(err_code);
    }

    template<typename ElemType>
    bool
    cuda_free(ElemType ptr)
    {
        using internal::cuda_free;
        using internal::cuda_error_t;

        cuda_error_t err_code = cuda_free(ptr);

        if(err_code != cuda_error_t::cuda_success)
            return false;
        return true;
    }
}
