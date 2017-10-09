
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/operations/mult.hpp>
#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/exception.hpp>

#include <cublas_v2.h>

namespace mgcpp
{
    template<size_t Device>
    gpu::matrix<float, Device, row_major>
    mult(gpu::matrix<float, Device, row_major> const& first,
         gpu::matrix<float, Device, row_major> const& second)
    {
        float const alpha = 1;
        float const beta = 0;

        auto second_shape = second.shape();
        auto first_shape = first.shape();

        size_t m = first_shape.first;
        size_t k = first_shape.second;
        size_t n = second_shape.second;

        gpu::matrix<float, Device, row_major> result{m, n};

        thread_context* context = first.get_thread_context();

        std::error_code status =
            cublasSgemm(context->get_cublas_context(Device),
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        m, n, k,
                        &alpha,
                        first.get_data(), m,
                        second.get_data(), k,
                        &beta,
                        result.get_data_mutable(), m);

        if(status != status_t::success)
            MGCPP_THROW_SYSTEM_ERROR(status);

        return result;
    }
}
