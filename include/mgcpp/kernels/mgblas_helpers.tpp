
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/kernels/bits/fill.cuh>
#include <mgcpp/kernels/mgblas_helpers.hpp>
#include <mgcpp/system/error_code.hpp>

namespace mgcpp
{
    outcome::result<void>
    mgblas_fill(float* arr, float value, size_t n)
    {
        std::error_code status = mgblas_Sfill(arr, value, n); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }

    outcome::result<void>
    mgblas_fill(double* arr, double value, size_t n)
    {
        std::error_code status = mgblas_Dfill(arr, value, n); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();

    }
}
