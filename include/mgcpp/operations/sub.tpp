//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda_libs/cublas.hpp>
#include <mgcpp/operations/sub.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    template<typename LhsDeviceVec,
             typename RhsDeviceVec,
             typename Type,
             size_t DeviceId>
    decltype(auto)
    strict::
    sub(dense_vector<LhsDeviceVec, Type, DeviceId> const& lhs,
        dense_vector<RhsDeviceVec, Type, DeviceId> const& rhs)
    {
        using allocator_type = typename LhsDeviceVec::allocator_type;
        using value_type = typename LhsDeviceVec::value_type;

        auto const& lhs_vec = ~lhs;
        auto const& rhs_vec = ~rhs;

        MGCPP_ASSERT(lhs_vec.shape() == rhs_vec.shape(),
                     "vector dimensions didn't match");

        auto* thread_context = lhs_vec.context();
        auto handle = thread_context->get_cublas_context(DeviceId);

        auto size = lhs_vec.shape();

        value_type const alpha = -1;

        auto result = device_vector<Type,
                                    DeviceId,
                                    allocator_type>(lhs_vec);
        auto status = cublas_axpy(handle, size,
                                  &alpha,
                                  rhs_vec.data(), 1,
                                  result.data_mutable(), 1);
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }

    template<typename LhsDenseMat,
             typename RhsDenseMat,
             typename Type,
             size_t DeviceId>
    decltype(auto)
    strict::
    sub(dense_matrix<LhsDenseMat, Type, DeviceId> const& lhs,
        dense_matrix<RhsDenseMat, Type, DeviceId> const& rhs)
    {
        using allocator_type = typename LhsDenseMat::allocator_type;
        using value_type = typename LhsDenseMat::value_type;

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

        size_t m = shape[0];
        size_t n = shape[1];

        value_type const alpha = 1;
        value_type const beta = -1;

        auto result = device_matrix<Type, DeviceId, allocator_type>({m, n});

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
