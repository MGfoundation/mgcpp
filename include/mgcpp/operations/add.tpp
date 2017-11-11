
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cublas/blaslike_ext.hpp>
#include <mgcpp/cublas/blas_lv1.hpp>
#include <mgcpp/operations/add.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/system/assert.hpp>

namespace mgcpp
{
    template<typename LhsMat, typename RhsMat, typename>
    device_matrix<typename LhsMat::value_type,
                  LhsMat::device_id,
                  typename LhsMat::allocator_type>
    strict::
    add(LhsMat const& first, RhsMat const& second)
    {
        using value_type = typename LhsMat::value_type;
        using allocator_type = typename LhsMat::allocator_type;
        size_t const device_id = LhsMat::device_id;

        MGCPP_ASSERT(first.shape() == second.shape(),
                         "matrix dimensions didn't match");

        auto set_device_status = cuda_set_device(LhsMat::device_id);
        if(!set_device_status)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

        auto* thread_context = first.context();
        auto handle = thread_context->get_cublas_context(device_id);

        auto shape = first.shape();

        auto m = shape.first;
        auto n = shape.second;

        value_type const alpha = 1;
        value_type const beta = 1;

        device_matrix<value_type,
                      device_id,
                      allocator_type> result{m, n};

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

    template<typename T, size_t Device, allignment Allign, typename Alloc>
    device_vector<T, Device, Allign, Alloc>
    strict::
    add(device_vector<T, Device, Allign, Alloc> const& first,
        device_vector<T, Device, Allign, Alloc> const& second)
    {
        MGCPP_ASSERT(first.shape() == second.shape(),
                     "vecotr size didn't match");

        device_vector<T, Device, Allign, Alloc> result(second);

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
