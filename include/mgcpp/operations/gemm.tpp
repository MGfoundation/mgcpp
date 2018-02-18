
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

namespace mgcpp {
template <typename ADense,
          typename BDense,
          typename CDense,
          typename Type,
          size_t DeviceId>
decltype(auto) strict::gemm(dense_matrix<ADense, Type, DeviceId> const& A,
                            dense_matrix<BDense, Type, DeviceId> const& B,
                            dense_matrix<CDense, Type, DeviceId> const& C) {
  return strict::gemm(Type(1), A, B, Type(B), C);
}

template <typename ADense,
          typename BDense,
          typename CDense,
          typename Type,
          size_t DeviceId,
          typename ScalarAlpha,
          typename ScalarBeta,
          typename>
decltype(auto) strict::gemm(ScalarAlpha alpha,
                            dense_matrix<ADense, Type, DeviceId> const& A,
                            dense_matrix<BDense, Type, DeviceId> const& B,
                            ScalarBeta beta,
                            dense_matrix<CDense, Type, DeviceId> const& C) {
  return strict::gemm(alpha, trans_mode::same, trans_mode::same, A, B, beta, C);
}

template <typename ADense,
          typename BDense,
          typename CDense,
          typename Type,
          size_t DeviceId,
          typename ScalarAlpha,
          typename ScalarBeta,
          typename>
decltype(auto) strict::gemm(ScalarAlpha alpha,
                            dense_matrix<ADense, Type, DeviceId> const& A,
                            dense_matrix<BDense, Type, DeviceId> const& B,
                            ScalarBeta beta,
                            dense_matrix<CDense, Type, DeviceId>&& C) {
  return strict::gemm(alpha, trans_mode::same, trans_mode::same, A, B, beta, std::move(C));
}

template <typename ADense,
          typename BDense,
          typename CDense,
          typename Type,
          size_t DeviceId,
          typename ScalarAlpha,
          typename ScalarBeta,
          typename>
inline decltype(auto) strict::gemm(
    ScalarAlpha alpha,
    trans_mode mode_A,
        trans_mode mode_B,
    dense_matrix<ADense, Type, DeviceId> const& A,
    dense_matrix<BDense, Type, DeviceId> const& B,
    ScalarBeta beta,
    dense_matrix<CDense, Type, DeviceId> const& C) {
  using device_pointer = typename ADense::device_pointer;
  using allocator_type = typename ADense::allocator_type;

  auto const& A_mat = ~A;
  auto const& B_mat = ~B;
  auto const& C_mat = ~C;

  auto A_shape = A_mat.shape();
  auto B_shape = B_mat.shape();
  auto C_shape = C_mat.shape();

  auto A_shape_after_trans = A_shape;
  if (mode_A == trans_mode::transposed || mode_A == trans_mode::conj_trans)
    std::swap(A_shape_after_trans[0], A_shape_after_trans[1]);
  auto B_shape_after_trans = B_shape;
  if (mode_B == trans_mode::transposed || mode_B == trans_mode::conj_trans)
    std::swap(B_shape_after_trans[0], B_shape_after_trans[1]);

  MGCPP_ASSERT(A_shape_after_trans[1] == B_shape_after_trans[0],
               "multiplied matrices' dimensions didn't match");

  MGCPP_ASSERT(C_mat.shape()[0] == A_shape_after_trans[0] &&
                   C_mat.shape()[1] == B_shape_after_trans[1],
               "added matrix' dimension doesn't match");

  auto set_device_status = cuda_set_device(DeviceId);
  if (!set_device_status) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_status.error());
  }

  auto* context = A_mat.context();
  auto handle = context->get_cublas_context(DeviceId);

  size_t m = A_shape_after_trans[0];
  size_t k = A_shape_after_trans[1];
  size_t n = B_shape_after_trans[1];

  auto result = device_matrix<Type, DeviceId, allocator_type>(C_mat);

  auto casted_alpha = Type(alpha);
  auto casted_beta = Type(beta);
  auto status = cublas_gemm(handle, static_cast<cublasOperation_t>(mode_A),
                            static_cast<cublasOperation_t>(mode_B), m, n, k,
                            pun_cast<device_pointer>(&casted_alpha),
                            A_mat.data(), A_shape[0], B_mat.data(), B_shape[0],
                            pun_cast<device_pointer>(&casted_beta),
                            result.data_mutable(), C_shape[0]);

  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  return result;
}

template <typename ADense,
          typename BDense,
          typename CDense,
          typename Type,
          size_t DeviceId,
          typename ScalarAlpha,
          typename ScalarBeta,
          typename>
inline decltype(auto) strict::gemm(
    ScalarAlpha alpha,
    trans_mode mode_A,
        trans_mode mode_B,
    dense_matrix<ADense, Type, DeviceId> const& A,
    dense_matrix<BDense, Type, DeviceId> const& B,
    ScalarBeta beta,
    dense_matrix<CDense, Type, DeviceId>&& C) {
  using device_pointer = typename ADense::device_pointer;

  auto const& A_mat = ~A;
  auto const& B_mat = ~B;
  auto C_mat = std::move(*static_cast<CDense*>(&C));

  auto A_shape = A_mat.shape();
  auto B_shape = B_mat.shape();
  auto C_shape = C_mat.shape();

  auto A_shape_after_trans = A_shape;
  if (mode_A == trans_mode::transposed || mode_A == trans_mode::conj_trans)
    std::swap(A_shape_after_trans[0], A_shape_after_trans[1]);
  auto B_shape_after_trans = B_shape;
  if (mode_B == trans_mode::transposed || mode_B == trans_mode::conj_trans)
    std::swap(B_shape_after_trans[0], B_shape_after_trans[1]);
  MGCPP_ASSERT(A_mat.shape()[1] == B_mat.shape()[0],
               "multiplied matrices' dimensions didn't match");

  MGCPP_ASSERT(C_mat.shape()[0] == A_mat.shape()[0] &&
                   C_mat.shape()[1] == B_mat.shape()[1],
               "added matrix' dimension doesn't match");

  auto set_device_status = cuda_set_device(DeviceId);
  if (!set_device_status) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_status.error());
  }

  auto* context = A_mat.context();
  auto handle = context->get_cublas_context(DeviceId);

  size_t m = A_shape_after_trans[0];
  size_t k = A_shape_after_trans[1];
  size_t n = B_shape_after_trans[1];

  auto casted_alpha = Type(alpha);
  auto casted_beta = Type(beta);
  auto status = cublas_gemm(handle, static_cast<cublasOperation_t>(mode_A),
                            static_cast<cublasOperation_t>(mode_B), m, n, k,
                            pun_cast<device_pointer>(&casted_alpha),
                            A_mat.data(), A_shape[0], B_mat.data(), B_shape[0],
                            pun_cast<device_pointer>(&casted_beta),
                            C_mat.data_mutable(), C_shape[0]);

  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  return C_mat;
}
}  // namespace mgcpp
