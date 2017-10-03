
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/operations/mult.hpp>
#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/exception.hpp>

#include <cublas_v2.h>

namespace mgcpp
{
    template<size_t Device>
    gpu::matrix
    mult(gpu::matrix<float, Device, row_major> const& first,
         gpu::matrix<float, Device, row_major> const& second)
    {
        float const alpha = 1;
        float const beta = 0;

        size_t m = second.rows(); // this is right to be flipped
        size_t k = second.columns();
        size_t n = first.columns();

        gpu::matrix<float, Device, row_major> result{};

        thread_context* context =  first.get_thread_context();
        
        std::error_code status =
            cublasSgemm_v2(context->get_cublas(Device),
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           m, n, k,
                           &alpha,
                           second.get_data(), m,
                           first.get_data(), k,
                           &beta,
                           result.get_data());

        if(status != status_t::success)
            MGCPP_THROW_SYSTEM_ERROR(status);

        return result;
    }
}
