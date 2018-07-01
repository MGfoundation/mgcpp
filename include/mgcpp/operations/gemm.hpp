
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_GEMM_HPP_
#define _MGCPP_OPERATIONS_GEMM_HPP_

#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/type_traits/is_scalar.hpp>

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
template <typename ADense, typename BDense, typename CDense, typename Type>
inline decltype(auto) gemm(dense_matrix<ADense, Type> const& A,
                           dense_matrix<BDense, Type> const& B,
                           dense_matrix<CDense, Type> const& C);

/**
 * General Matrix-matrix Multiplication.
 * \param alpha scalar constant to multiply to A
 * \param A left-hand side matrix operand
 * \param B right-hand side matrix operand
 * \param beta scalar constant to multiply to C
 * \param C matrix to add after the multiplication
 * \returns alpha * A * B + beta * C
 */
template <
    typename ADense,
    typename BDense,
    typename CDense,
    typename Type,
    typename ScalarAlpha,
    typename ScalarBeta,
    typename = typename std::enable_if<is_scalar<ScalarAlpha>::value &&
                                       is_scalar<ScalarBeta>::value>::type>
inline decltype(auto) gemm(ScalarAlpha alpha,
                           dense_matrix<ADense, Type> const& A,
                           dense_matrix<BDense, Type> const& B,
                           ScalarBeta beta,
                           dense_matrix<CDense, Type> const& C);

/**
 * General Matrix-matrix Multiplication.
 * \param alpha scalar constant to multiply to A
 * \param A left-hand side matrix operand
 * \param B right-hand side matrix operand
 * \param beta scalar constant to multiply to C
 * \param C matrix to add after the multiplication. It is moved instead of
 * copied.
 * \returns alpha * A * B + beta * C
 */
template <
    typename ADense,
    typename BDense,
    typename CDense,
    typename Type,
    typename ScalarAlpha,
    typename ScalarBeta,
    typename = typename std::enable_if<is_scalar<ScalarAlpha>::value &&
                                       is_scalar<ScalarBeta>::value>::type>
inline decltype(auto) gemm(ScalarAlpha alpha,
                           dense_matrix<ADense, Type> const& A,
                           dense_matrix<BDense, Type> const& B,
                           ScalarBeta beta,
                           dense_matrix<CDense, Type>&& C);

enum class trans_mode {
  same = CUBLAS_OP_N,
  transposed = CUBLAS_OP_T,
  conj_trans = CUBLAS_OP_C
};

/**
 * General Matrix-matrix Multiplication.
 * \param alpha scalar constant to multiply to A
 * \param mode_A transposition mode to apply to A before the operation.
 * Can be strict::trans_mode::same, strict::trans_mode::transposed, or
 * strict::trans_mode::conj_trans.
 * \param A left-hand side matrix operand
 * \param B right-hand side matrix operand
 * \param beta scalar constant to multiply to C
 * \param mode_B transposition mode to apply to B before the operation. The
 * accepted values are the same as mode_A.
 * \param C matrix to add after the multiplication.
 * \returns alpha * mode_A(A) * mode_B(B) + beta * C
 */
template <
    typename ADense,
    typename BDense,
    typename CDense,
    typename Type,
    typename ScalarAlpha,
    typename ScalarBeta,
    typename = typename std::enable_if<is_scalar<ScalarAlpha>::value &&
                                       is_scalar<ScalarBeta>::value>::type>
inline decltype(auto) gemm(ScalarAlpha alpha,
                           trans_mode mode_A,
                           trans_mode mode_B,
                           dense_matrix<ADense, Type> const& A,
                           dense_matrix<BDense, Type> const& B,
                           ScalarBeta beta,
                           dense_matrix<CDense, Type> const& C);

/**
 * General Matrix-matrix Multiplication.
 * \param alpha scalar constant to multiply to A
 * \param mode_A transposition mode to apply to A before the operation.
 * Can be strict::trans_mode::same, strict::trans_mode::transposed, or
 * strict::trans_mode::conj_trans.
 * \param A left-hand side matrix operand
 * \param B right-hand side matrix operand
 * \param beta scalar constant to multiply to C
 * \param mode_B transposition mode to apply to B before the operation. The
 * accepted values are the same as mode_A.
 * \param C matrix to add after the multiplication. It is moved instead of
 * copied.
 * \returns alpha * mode_A(A) * mode_B(B) + beta * C
 */
template <
    typename ADense,
    typename BDense,
    typename CDense,
    typename Type,
    typename ScalarAlpha,
    typename ScalarBeta,
    typename = typename std::enable_if<is_scalar<ScalarAlpha>::value &&
                                       is_scalar<ScalarBeta>::value>::type>
inline decltype(auto) gemm(ScalarAlpha alpha,
                           trans_mode mode_A,
                           trans_mode mode_B,
                           dense_matrix<ADense, Type> const& A,
                           dense_matrix<BDense, Type> const& B,
                           ScalarBeta beta,
                           dense_matrix<CDense, Type>&& C);
}  // namespace strict
}  // namespace mgcpp

#include <mgcpp/operations/gemm.tpp>
#endif
