
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/cublas/blas_lv1.hpp>
#include <mgcpp/cublas/blas_lv3.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/system/assert.hpp>

namespace mgcpp
{
    template<typename LhsDenseMat,
             typename RhsDenseMat,
             typename Type,
             size_t DeviceId>
    device_matrix<Type, DeviceId, typename LhsDenseMat::allocator_type>
    strict::
    mult(dense_matrix<LhsDenseMat, Type, DeviceId> const& lhs,
         dense_matrix<RhsDenseMat, Type, DeviceId> const& rhs)
    {
        using allocator_type = typename LhsDenseMat::allocator_type;

        auto const& lhs_mat = ~lhs;
        auto const& rhs_mat = ~rhs;

        MGCPP_ASSERT(lhs_mat.shape().second == rhs_mat.shape().first,
                     "matrix dimensions didn't match");

        auto set_device_status = cuda_set_device(DeviceId);
        if(!set_device_status)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

        auto* context = lhs_mat.context();
        auto handle = context->get_cublas_context(DeviceId);

        auto lhs_shape = lhs_mat.shape();
        auto rhs_shape = rhs_mat.shape();

        size_t m = lhs_shape.first;
        size_t k = lhs_shape.second;
        size_t n = rhs_shape.second;

        Type const alpha = 1;
        Type const beta = 0;

        auto result = device_matrix<Type, DeviceId, allocator_type>{m, n};

        auto status = cublas_gemm(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, n, k,
                                  &alpha,
                                  lhs_mat.data(), m,
                                  rhs_mat.data(), k,
                                  &beta,
                                  result.data_mutable(), m);

        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }
    
    template<typename DenseVec,
             typename Type,
             alignment Align,
             size_t DeviceId>
    device_vector<Type, Align, DeviceId, 
                  typename DenseVec::allocator_type>
    strict::
    mult(Type scalar,
         dense_vector<DenseVec, Type, Align, DeviceId> const& vec)
    {
        using allocator_type = typename DenseVec::allocator_type;

        auto const& original_vec = ~vec;

        auto* context = original_vec.context();
        auto handle = context->get_cublas_context(DeviceId);

        auto size = original_vec.shape();

        auto result = device_vector<Type,
                                    Align,
                                    DeviceId,
                                    allocator_type>(original_vec);
        auto status = cublas_scal(handle, size,
                                  &scalar,
                                  result.data_mutable(), 1);
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }
}
