
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cuda_runtime.h>

#include <mgcpp/cuda/stdlib.hpp>
#include <mgcpp/system/error_code.hpp>
#include <mgcpp/cuda/exception.hpp>

namespace mgcpp
{
    template<typename ElemType, typename>
    ElemType*
    cuda_malloc(size_t size)
    {
        void* ptr = nullptr;
        std::error_code err_code =
            cudaMalloc(&ptr, size * sizeof(ElemType));

        if(err_code != make_error_condition(status_t::success))
            MGCPP_THROW_BAD_ALLOC;

        return static_cast<ElemType*>(ptr);
    }

    template<typename ElemType, typename>
    ElemType*
    cuda_malloc(size_t size,
                std::nothrow_t const& nothrow_flag) noexcept
    {
        (void)nothrow_flag; // warning suppression

        void* ptr = nullptr;
        std::error_code err_code =
            cudaMalloc(&ptr, size * sizeof(ElemType));

        if(err_code != make_error_condition(status_t::success))
            return nullptr;

        return static_cast<ElemType*>(ptr);
    }

    template<typename ElemType>
    bool
    cuda_free(ElemType* ptr) noexcept
    {
        std::error_code err_code = cudaFree(ptr);

        if(err_code != make_error_condition(status_t::success))
            return false;
        return true;
    }
}
