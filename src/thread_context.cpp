
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    cublasHandle_t
    thread_context::
    get_cublas_context(size_t device_id) 
    {
        auto& handle = _cublas_handle[device_id];

        if(!handle)
        {
            auto set_device_stat = cuda_set_device(device_id);
            if(!set_device_stat)
                MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());

            cublasHandle_t new_handle;

            std::error_code status = cublasCreate(&new_handle);
            if(status != status_t::success)
                MGCPP_THROW_SYSTEM_ERROR(status);

            handle = cublas_handle_unique_ptr(
                new_handle,
                [device_id](cublasHandle_t handle)
                {
                    cuda_set_device(device_id);
                    cublasDestroy(handle);
                });
        }

        return handle.get();
    }

    thread_context::
    thread_context(thread_context&& other) noexcept
        : _cublas_handle(std::move(other._cublas_handle))
    {}

    thread_context&
    thread_context::
    operator=(thread_context&& other) noexcept
    {
        _cublas_handle = std::move(other._cublas_handle);
        return *this;
    }
}
