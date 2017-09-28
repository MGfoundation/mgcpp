
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/device_manager.hpp>
#include <mgcpp/system/error_code.hpp>


namespace mgcpp
{
    cublasHandle_t
    device_manager::
    create_cublas_handle() const
    {
        cublasHandle_t handle; 
        std::error_code status = cublasCreate(&handle);

        if(status != make_error_condition(status_t::success))
            return handle;

        return handle;
    }

    cublasHandle_t
    device_manager::
    get_cublas() 
    {
        if(!_cublas_handle)
            _cublas_handle = create_cublas_handle();
        return *_cublas_handle;
    }
}
