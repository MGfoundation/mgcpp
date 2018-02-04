
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/cuda_libs/cublas.hpp>
#include <mgcpp/operations/gemm.hpp>
#include <mgcpp/system/assert.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/system/pun_cast.hpp>

namespace mgcpp
{
    template<typename ADense,
             typename BDense,
             typename CDense,
             typename Type,
             size_t DeviceId>
    decltype(auto)
    strict::
    gemm(dense_matrix<ADense, Type, DeviceId> const& A,
         dense_matrix<BDense, Type, DeviceId> const& B,
         dense_matrix<CDense, Type, DeviceId> const& C)
    {
        using allocator_type = typename ADense::allocator_type;

        auto const& A_mat = ~A;
        auto const& B_mat = ~B;
        auto const& C_mat = ~C;

        MGCPP_ASSERT(A_mat.shape()[1] == B_mat.shape()[0],
                     "multiplied matrices' dimensions didn't match");

        MGCPP_ASSERT(C_mat.shape()[0] == A_mat.shape()[0]
                     && C_mat.shape()[1] == B_mat.shape()[1],
                     "added matrix' dimension doesn't match");

        auto set_device_status = cuda_set_device(DeviceId);
        if(!set_device_status)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

        auto* context = A_mat.context();
        auto handle = context->get_cublas_context(DeviceId);

        auto A_shape = A_mat.shape();
        auto B_shape = B_mat.shape();

        size_t m = A_shape[0];
        size_t k = A_shape[1];
        size_t n = B_shape[1];

        Type const alpha = Type(1);
        Type const beta = Type(1);

        auto result = device_matrix<Type, DeviceId, allocator_type>(C_mat);

        auto status = cublas_gemm(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, n, k,
                                  &alpha,
                                  A_mat.data(), m,
                                  B_mat.data(), k,
                                  &beta,
                                  result.data_mutable(), m);

        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }

    template<typename ADense,
             typename BDense,
             typename CDense,
             typename Type,
             size_t DeviceId,
             typename ScalarAlpha,
             typename ScalarBeta,
             typename>
    decltype(auto)
    strict::
    gemm(ScalarAlpha alpha,
         dense_matrix<ADense, Type, DeviceId> const& A,
         dense_matrix<BDense, Type, DeviceId> const& B,
         ScalarBeta beta,
         dense_matrix<CDense, Type, DeviceId> const& C)
    {
        using device_pointer = typename ADense::device_pointer;
        using allocator_type = typename ADense::allocator_type;

        auto const& A_mat = ~A;
        auto const& B_mat = ~B;
        auto const& C_mat = ~C;

        MGCPP_ASSERT(A_mat.shape()[1] == B_mat.shape()[0],
                     "multiplied matrices' dimensions didn't match");

        MGCPP_ASSERT(C_mat.shape()[0] == A_mat.shape()[0]
                     && C_mat.shape()[1] == B_mat.shape()[1],
                     "added matrix' dimension doesn't match");

        auto set_device_status = cuda_set_device(DeviceId);
        if(!set_device_status)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

        auto* context = A_mat.context();
        auto handle = context->get_cublas_context(DeviceId);

        auto A_shape = A_mat.shape();
        auto B_shape = B_mat.shape();

        size_t m = A_shape[0];
        size_t k = A_shape[1];
        size_t n = B_shape[1];

        auto result = device_matrix<Type, DeviceId, allocator_type>(C_mat);

        auto casted_alpha = Type(alpha);
        auto casted_beta = Type(beta);
        auto status = cublas_gemm(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, n, k,
                                  pun_cast<device_pointer>(&casted_alpha),
                                  A_mat.data(), m,
                                  B_mat.data(), k,
                                  pun_cast<device_pointer>(&casted_beta),
                                  result.data_mutable(), m);

        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }

    template<typename ADense,
             typename BDense,
             typename CDense,
             typename Type,
             size_t DeviceId,
             typename ScalarAlpha,
             typename ScalarBeta,
             typename>
    decltype(auto)
    strict::
    gemm(ScalarAlpha alpha,
         dense_matrix<ADense, Type, DeviceId> const& A,
         dense_matrix<BDense, Type, DeviceId> const& B,
         ScalarBeta beta,
         dense_matrix<CDense, Type, DeviceId>&& C)
    {
        using device_pointer = typename ADense::device_pointer;

        auto const& A_mat = ~A;
        auto const& B_mat = ~B;
        auto&& C_mat = std::move(*static_cast<CDense*>(&C));

        MGCPP_ASSERT(A_mat.shape()[1] == B_mat.shape()[0],
                     "multiplied matrices' dimensions didn't match");

        MGCPP_ASSERT(C_mat.shape()[0] == A_mat.shape()[0]
                     && C_mat.shape()[1] == B_mat.shape()[1],
                     "added matrix' dimension doesn't match");

        auto set_device_status = cuda_set_device(DeviceId);
        if(!set_device_status)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

        auto* context = A_mat.context();
        auto handle = context->get_cublas_context(DeviceId);

        auto A_shape = A_mat.shape();
        auto B_shape = B_mat.shape();

        size_t m = A_shape[0];
        size_t k = A_shape[1];
        size_t n = B_shape[1];
        
        auto casted_alpha = Type(alpha);
        auto casted_beta = Type(beta);
        auto status = cublas_gemm(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, n, k,
                                  pun_cast<device_pointer>(&casted_alpha),
                                  A_mat.data(), m,
                                  B_mat.data(), k,
                                  pun_cast<device_pointer>(&casted_beta),
                                  C_mat.data_mutable(), m);

        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return C_mat;
    }
}
