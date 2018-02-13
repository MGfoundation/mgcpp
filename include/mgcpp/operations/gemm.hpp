
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_GEMM_HPP_
#define _MGCPP_OPERATIONS_GEMM_HPP_

#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/type_traits/type_traits.hpp>

#include <cstdlib>

namespace mgcpp {
namespace strict {
/**
 * General Matrix-matrix Multiplication.
 * \param A left-hand side matrix operand
 * \param B right-hand side matrix operand
 * \param C matrix to add after the multiplication
 * \returns A * B + C
 */
template <typename ADense,
          typename BDense,
          typename CDense,
          typename Type,
          size_t DeviceId>
inline decltype(auto) gemm(dense_matrix<ADense, Type, DeviceId> const& A,
                           dense_matrix<BDense, Type, DeviceId> const& B,
                           dense_matrix<CDense, Type, DeviceId> const& C);

/**
 * General Matrix-matrix Multiplication.
 * \param alpha scalar constant to multiply to A
 * \param A left-hand side matrix operand
 * \param B right-hand side matrix operand
 * \param beta scalar constant to multiply to B
 * \param C matrix to add after the multiplication
 * \returns alpha * A * B + beta * C
 */
template <
    typename ADense,
    typename BDense,
    typename CDense,
    typename Type,
    size_t DeviceId,
    typename ScalarAlpha,
    typename ScalarBeta,
    typename = typename std::enable_if<is_scalar<ScalarAlpha>::value &&
                                       is_scalar<ScalarBeta>::value>::type>
inline decltype(auto) gemm(ScalarAlpha alpha,
                           dense_matrix<ADense, Type, DeviceId> const& A,
                           dense_matrix<BDense, Type, DeviceId> const& B,
                           ScalarBeta beta,
                           dense_matrix<CDense, Type, DeviceId> const& C);

/**
 * General Matrix-matrix Multiplication.
 * \param alpha scalar constant to multiply to A
 * \param A left-hand side matrix operand
 * \param B right-hand side matrix operand
 * \param beta scalar constant to multiply to B
 * \param C matrix to add after the multiplication. It is moved instead of
 * copying. \returns alpha * A * B + beta * C
 */
template <
    typename ADense,
    typename BDense,
    typename CDense,
    typename Type,
    size_t DeviceId,
    typename ScalarAlpha,
    typename ScalarBeta,
    typename = typename std::enable_if<is_scalar<ScalarAlpha>::value &&
                                       is_scalar<ScalarBeta>::value>::type>
inline decltype(auto) gemm(ScalarAlpha alpha,
                           dense_matrix<ADense, Type, DeviceId> const& A,
                           dense_matrix<BDense, Type, DeviceId> const& B,
                           ScalarBeta beta,
                           dense_matrix<CDense, Type, DeviceId>&& C);
}  // namespace strict
}  // namespace mgcpp

#include <mgcpp/operations/gemm.tpp>
#endif
