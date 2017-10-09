
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/exception.hpp>

#include <iostream>

namespace mgcpp
{
    cublasHandle_t 
    thread_context::
    get_cublas_context(size_t device_id) 
    {
        auto& handle = _cublas_handle[device_id];
        if(!handle)
        {
            cublasHandle_t new_handle;

            std::error_code status = cublasCreate(&new_handle);
            std::cout << "safe?" << std::endl;
            std::cout << "handle: " << (int*)new_handle;
            std::cout << "safe?" << std::endl;

            if(status != status_t::success)
                MGCPP_THROW_SYSTEM_ERROR(status);

            handle = cublas_handle_unique_ptr(
                &new_handle,
                [](cublasHandle_t* handle)
                {
                    std::cout << "destroying handle: " << (int*)handle;
                    cublasDestroy(*handle);
                });
        }

        return *handle;
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
