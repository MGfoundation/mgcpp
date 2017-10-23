
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/operations/add.hpp>
#include <mgcpp/cublas/blaslike_ext.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    template<typename T, size_t Device, storage_order SO>
    gpu::matrix<T, Device, SO>
    strict::
    add(gpu::matrix<T, Device, SO> const& first,
        gpu::matrix<T, Device, SO> const& second)
    {
        auto* thread_context = first.get_thread_context();
        auto handle = thread_context->get_cublas_context(Device);

        auto shape = first.shape();

        auto m = shape.first;
        auto n = shape.second;

        T const alpha = 1;
        T const beta = 1;

        gpu::matrix<T, Device, SO> result{m, n};

        auto status = cublas_geam(handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  m, n,
                                  &alpha,
                                  first.get_data(), m,
                                  &beta,
                                  second.get_data(), m,
                                  result.get_data_mutable(), m);

        if(!status)
            MGCPP_THROW_SYSTEM_ERROR(status.error());

        return result;
    }
}
