
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cuda_runtime.h>

#include <mgcpp/cuda/memory.hpp>
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

    template<typename ElemType>
    outcome::result<void>
    cuda_memset(ElemType* ptr, ElemType value, size_t count) noexcept
    {
        std::error_code err_code =
            cudaMemset((void*)ptr, value, count);

        if(err_code != make_error_condition(status_t::success))
            return err_code;

        return outcome::success();
    }

    outcome::result<std::pair<free_mem_t, total_mem_t>>
    cuda_mem_get_info() noexcept
    {
        size_t free_memory;
        size_t total_memory;

        std::error_code status =
            cudaMemGetInfo(&free_memory, &total_memory);

        if(status != make_error_condition(status_t::success))
            return status;
        else
            return std::make_pair(free_memory, total_memory);
    }
}
