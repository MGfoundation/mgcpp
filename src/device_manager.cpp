
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/device_manager.hpp>
#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    device_manager::
    device_manager(size_t device_id) :_device_id(device_id) {}

    device_manager::
    device_manager(device_manager&& other) noexcept
        : _device_id(other._device_id),
          _cublas_handle(std::move(other._cublas_handle))
    {
        other._cublas_handle = nullptr;
    }

    device_manager&
    device_manager::
    operator=(device_manager&& other) noexcept
    {
        _cublas_handle = std::move(other._cublas_handle);
        _device_id = other._device_id;

        return *this;
    }

    std::unique_ptr<cublasHandle_t>
    device_manager::
    create_cublas_handle() const
    {
        cublasHandle_t handle; 
        std::error_code status = cublasCreate(&handle);

        if(status != status_t::success)
            MGCPP_THROW_SYSTEM_ERROR(status);

        return std::make_unique<cublasHandle_t>(handle);
    }

    cublasHandle_t
    device_manager::
    get_cublas() 
    {
        if(!_cublas_handle)
            _cublas_handle = create_cublas_handle();
        return *_cublas_handle;
    }

    device_manager::
    ~device_manager()
    {
        cublasDestroy(*_cublas_handle);
    }
}
