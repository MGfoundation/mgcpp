
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/cuda_libs/cublas.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/system/assert.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/system/pun_cast.hpp>

namespace mgcpp
{
    template<typename LhsDenseMat,
             typename RhsDenseMat,
             typename Type,
             size_t DeviceId>
    decltype(auto)
    strict::
    mult(dense_matrix<LhsDenseMat, Type, DeviceId> const& lhs,
         dense_matrix<RhsDenseMat, Type, DeviceId> const& rhs)
    {
        using allocator_type = typename LhsDenseMat::allocator_type;

        auto const& lhs_mat = ~lhs;
        auto const& rhs_mat = ~rhs;

        MGCPP_ASSERT(lhs_mat.shape()[1] == rhs_mat.shape()[0],
                     "matrix dimensions didn't match");

        auto set_device_status = cuda_set_device(DeviceId);
        if(!set_device_status)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

        auto* context = lhs_mat.context();
        auto handle = context->get_cublas_context(DeviceId);

        auto lhs_shape = lhs_mat.shape();
        auto rhs_shape = rhs_mat.shape();

        size_t m = lhs_shape[0];
        size_t k = lhs_shape[1];
        size_t n = rhs_shape[1];

        Type const alpha = 1;
        Type const beta = 0;

        auto result = device_matrix<Type, DeviceId, allocator_type>({m, n});

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
    
    template<typename DenseMat,
             typename DenseVec,
             typename Type,
             alignment Align,
             size_t DeviceId>
    inline decltype(auto)
    strict::
    mult(dense_matrix<DenseMat, Type, DeviceId> const& mat,
         dense_vector<DenseVec, Type, Align, DeviceId> const& vec)
    {
        using allocator_type = typename DenseVec::allocator_type;

        auto const& dmat = ~mat;
        auto const& dvec = ~vec;

        auto *context = dmat.context();
        auto handle = context->get_cublas_context(DeviceId);

        MGCPP_ASSERT(dmat.shape()[1] == dvec.shape(),
                     "Matrix.shape[1] != Vector.shape");

        auto n = dmat.shape()[0];
        auto k = dmat.shape()[1];

        auto result = device_vector<Type, Align, DeviceId, allocator_type>(n);

        Type const alpha = 1;
        Type const beta = 0;

        auto status = cublas_gemv(handle,
                                  CUBLAS_OP_N,
                                  n, k,
                                  &alpha,
                                  dmat.data(), n,
                                  dvec.data(), 1,
                                  &beta,
                                  result.data_mutable(), 1);

        if (!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }

    template<typename DenseVec,
             typename ScalarType,
             typename VectorType,
             alignment Align,
             size_t DeviceId,
             typename>
    decltype(auto)
    strict::
    mult(ScalarType scalar,
         dense_vector<DenseVec, VectorType, Align, DeviceId> const& vec)
    {
        using allocator_type = typename DenseVec::allocator_type;
        using device_pointer = typename DenseVec::device_pointer;

        auto const& original_vec = ~vec;

        auto* context = original_vec.context();
        auto handle = context->get_cublas_context(DeviceId);

        auto size = original_vec.shape();

        // complex scalar x real vector will need something
        auto casted_scalar = VectorType(scalar);
        auto result = device_vector<VectorType,
                                    Align,
                                    DeviceId,
                                    allocator_type>(original_vec);
        auto status = cublas_scal(handle, size,
                                  pun_cast<device_pointer>(&casted_scalar),
                                  result.data_mutable(), 1);
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }


    template<typename DenseMat,
             typename MatrixType,
             size_t DeviceId,
             typename ScalarType,
             typename>
    inline decltype(auto)
    strict::
    mult(ScalarType scalar,
         dense_matrix<DenseMat, MatrixType, DeviceId> const& mat)
    {
        using allocator_type = typename DenseMat::allocator_type;
        using device_pointer = typename DenseMat::device_pointer;

        auto const& original_mat = ~mat;

        auto* context = original_mat.context();
        auto handle = context->get_cublas_context(DeviceId);

        auto size = original_mat.shape();

        // complex scalar x real matrix will need something
        auto casted_scalar = MatrixType(scalar);
        auto result = device_matrix<MatrixType,
                                    DeviceId,
                                    allocator_type>(original_mat);
        auto status = cublas_scal(handle, size[0] * size[1],
                                  pun_cast<device_pointer>(&casted_scalar),
                                  result.data_mutable(), 1);
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }
}
