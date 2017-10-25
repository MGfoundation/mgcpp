
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
        auto* thread_context = first.context();
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
                                  first.data(), m,
                                  &beta,
                                  second.data(), m,
                                  result.data_mutable(), m);

        if(!status)
            MGCPP_THROW_SYSTEM_ERROR(status.error());

        return result;
    }

    template<typename T, size_t Device, allignment Allign>
    gpu::vector<T, Device, Allign>
    strict::
    add(gpu::vector<T, Device, Allign> const& first,
        gpu::vector<T, Device, Allign> const& second)
    {
        gpu::vector<T, Device, Allign> result(second);

        auto* thread_context = first.context();
        auto handle = thread_context->get_cublas_context(Device);

        T const alpha = 1;
        auto size = first.shape();

        auto status = cublas_axpy(handle, size,
                                  &alpha,
                                  first.data(), 1,
                                  result.data_mutable(), 1);
        if(!status)
        {
            MGCPP_THROW_SYSTEM_ERROR(status.error());
        }

        return result;
    }
}
