
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cublas/blaslike_ext.hpp>
#include <mgcpp/operations/sub.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    template<typename T, size_t Device, allignment Allign, typename Alloc>
    device_vector<T, Device, Allign, Alloc>
    strict::
    sub(device_vector<T, Device, Allign, Alloc> const& first,
        device_vector<T, Device, Allign, Alloc> const& second)
    {
        MGCPP_ASSERT(first.shape() == second.shape(),
                     "vector dimensions didn't match");

        device_vector<T, Device, Allign, Alloc> result(first);

        auto* thread_context = first.context();
        auto handle = thread_context->get_cublas_context(Device);

        T const alpha = -1;

        auto size = first.shape();

        auto status = cublas_axpy(handle, size,
                                  &alpha,
                                  second.data(), 1,
                                  result.data_mutable(), 1);
        if(!status)
        {
            MGCPP_THROW_SYSTEM_ERROR(status.error());
        }

        return result;
    }

    template<typename LhsDenseMat,
             typename RhsDenseMat,
             typename Type,
             size_t DeviceId>
    device_matrix<Type, DeviceId, typename LhsDenseMat::allocator_type>
    strict::
    sub(dense_matrix<LhsDenseMat, Type, DeviceId> const& lhs,
        dense_matrix<RhsDenseMat, Type, DeviceId> const& rhs)
    {
        using allocator_type = typename LhsDenseMat::allocator_type;

        auto const& lhs_mat = ~lhs;
        auto const& rhs_mat = ~rhs;

        MGCPP_ASSERT(lhs_mat.shape() == rhs_mat.shape(),
                     "matrix dimensions didn't match");

        auto set_device_status = cuda_set_device(DeviceId);
        if(!set_device_status)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

        auto* thread_context = lhs_mat.context();
        auto handle = thread_context->get_cublas_context(DeviceId);

        auto shape = lhs_mat.shape();

        size_t m = shape.first;
        size_t n = shape.second;

        Type const alpha = 1;
        Type const beta = -1;

        auto result = device_matrix<Type, DeviceId, allocator_type>{m, n};

        auto status = cublas_geam(handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  m, n,
                                  &alpha,
                                  lhs_mat.data(), m,
                                  &beta,
                                  rhs_mat.data(), m,
                                  result.data_mutable(), m);

        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); } 

        return result;
    }
}
