
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_ABSOLUTE_HPP_
#define _MGCPP_OPERATIONS_ABSOLUTE_HPP_

#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

#include <cstdlib>

namespace mgcpp {
namespace strict {

template <typename Type,
          outcome::result<void> (*Function)(Type*, size_t),
          typename DenseVec,
          size_t DeviceId>
inline device_vector<Type, DeviceId, typename DenseVec::allocator_type>
map(dense_vector<DenseVec, Type, DeviceId> const& vec);

template <typename Type,
          outcome::result<void> (*Function)(Type*, size_t),
          typename DenseMat,
          size_t DeviceId>
inline device_matrix<Type, DeviceId, typename DenseMat::allocator_type>
map(dense_matrix<DenseMat, Type, DeviceId> const& mat);

/** Computes the absolute value of each element.
 * \param vec vector to calculate the absolute value of.
 */
template <typename DenseVec, typename Type, size_t DeviceId>
inline decltype(auto) abs(dense_vector<DenseVec, Type, DeviceId> const& vec);

/** Computes the absolute value of each element.
 * \param mat matrix to calculate the absolute value of.
 */
template <typename DenseMat, typename Type, size_t DeviceId>
inline decltype(auto) abs(dense_matrix<DenseMat, Type, DeviceId> const& mat);

template <typename DenseVec, typename Type, size_t DeviceId>
inline decltype(auto) sin(dense_vector<DenseVec, Type, DeviceId> const& vec);

template <typename DenseMat, typename Type, size_t DeviceId>
inline decltype(auto) sin(dense_matrix<DenseMat, Type, DeviceId> const& mat);

template <typename DenseVec, typename Type, size_t DeviceId>
inline decltype(auto) cos(dense_vector<DenseVec, Type, DeviceId> const& vec);

template <typename DenseMat, typename Type, size_t DeviceId>
inline decltype(auto) cos(dense_matrix<DenseMat, Type, DeviceId> const& mat);

template <typename DenseVec, typename Type, size_t DeviceId>
inline decltype(auto) tan(dense_vector<DenseVec, Type, DeviceId> const& vec);

template <typename DenseMat, typename Type, size_t DeviceId>
inline decltype(auto) tan(dense_matrix<DenseMat, Type, DeviceId> const& mat);

template <typename DenseVec, typename Type, size_t DeviceId>
inline decltype(auto) sinh(dense_vector<DenseVec, Type, DeviceId> const& vec);

template <typename DenseMat, typename Type, size_t DeviceId>
inline decltype(auto) sinh(dense_matrix<DenseMat, Type, DeviceId> const& mat);

template <typename DenseVec, typename Type, size_t DeviceId>
inline decltype(auto) cosh(dense_vector<DenseVec, Type, DeviceId> const& vec);

template <typename DenseMat, typename Type, size_t DeviceId>
inline decltype(auto) cosh(dense_matrix<DenseMat, Type, DeviceId> const& mat);

template <typename DenseVec, typename Type, size_t DeviceId>
inline decltype(auto) tanh(dense_vector<DenseVec, Type, DeviceId> const& vec);

template <typename DenseMat, typename Type, size_t DeviceId>
inline decltype(auto) tanh(dense_matrix<DenseMat, Type, DeviceId> const& mat);

template <typename DenseVec, typename Type, size_t DeviceId>
inline decltype(auto) relu(dense_vector<DenseVec, Type, DeviceId> const& vec);

template <typename DenseMat, typename Type, size_t DeviceId>
inline decltype(auto) relu(dense_matrix<DenseMat, Type, DeviceId> const& mat);
}  // namespace strict
}  // namespace mgcpp

#include <mgcpp/operations/map.tpp>
#endif
