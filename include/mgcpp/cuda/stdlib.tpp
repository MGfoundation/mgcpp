
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cuda_runtime.h>

#include <mgcpp/cuda/stdlib.hpp>
#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    template<typename ElemType, typename>
    outcome::result<ElemType*>
    cuda_malloc(size_t size) noexcept
    {
        void* ptr = nullptr;
        std::error_code err_code =
            cudaMalloc(&ptr, size * sizeof(ElemType));

        if(err_code != make_error_condition(status_t::success))
            return err_code;

        return static_cast<ElemType*>(ptr);
    }

    template<typename ElemType>
    outcome::result<void>
    cuda_free(ElemType* ptr) noexcept
    {
        std::error_code err_code = cudaFree(ptr);

        if(err_code != make_error_condition(status_t::success))
            return err_code;

        return outcome::success();
    }
}
