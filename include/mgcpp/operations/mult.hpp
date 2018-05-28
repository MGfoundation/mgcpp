
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_MULTIPLICATION_HPP_
#define _MGCPP_OPERATIONS_MULTIPLICATION_HPP_

#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/type_traits/is_scalar.hpp>
#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

#include <cstdlib>

namespace mgcpp {
namespace strict {
/**
 * Matrix-matrix multiplication.
 * \param lhs left-hand side matrix
 * \param rhs right-hand side matrix
 * \returns lhs * rhs
 */
template <typename LhsDenseMat,
          typename RhsDenseMat,
          typename Type>
inline decltype(auto) mult(
    dense_matrix<LhsDenseMat, Type> const& lhs,
    dense_matrix<RhsDenseMat, Type> const& rhs);

// template<typename LhsDenseVec,
//          typename RhsDenseVec,
//          typename Type,
//          size_t Device,
//          alignment Align>
// inline device_vector<Type, Device, Align,
//                      typename LhsDenseVec::allocator_type>
// mult(dense_vector<LhsDenseVec, Type, Device, Align> const& first,
//      dense_vector<RhsDenseVec, Type, Device, Align> const& second);

/**
 * Matrix-vector multiplication.
 * If matrix is size (n x m), then vector's size must be m.
 * The resulting vector is of size n.
 * \param mat left-hand side matrix
 * \param vec right-hand side vector
 * \returns mat * vec
 */
template <typename DenseMat, typename DenseVec, typename Type>
inline decltype(auto) mult(dense_matrix<DenseMat, Type> const& mat,
                           dense_vector<DenseVec, Type> const& vec);

/**
 * Scalar-vector multiplication.
 * \param scalar the scalar multiplier
 * \param vec vector to be multiplied by the scalar
 * \returns scalar * vec
 */
template <
    typename DenseVec,
    typename ScalarType,
    typename VectorType,
    typename = typename std::enable_if<is_scalar<ScalarType>::value>::type>
inline decltype(auto) mult(
    ScalarType scalar,
    dense_vector<DenseVec, VectorType> const& vec);

/**
 * Scalar-matrix multiplication.
 * \param scalar the scalar multiplier
 * \param mat matrix to be multiplied by the scalar
 * \returns scalar * mat
 */
template <
    typename DenseMat,
    typename MatrixType,
    typename ScalarType,
    typename = typename std::enable_if<is_scalar<ScalarType>::value>::type>
inline decltype(auto) mult(
    ScalarType scalar,
    dense_matrix<DenseMat, MatrixType> const& mat);

// template<typename T, size_t Device, storage_order SO>
// void
// mult_assign(gpu::matrix<T, Device, SO>& first,
//             gpu::matrix<T, Device, SO> const& second);
}  // namespace strict
}  // namespace mgcpp

#include <mgcpp/operations/mult.tpp>
#endif
