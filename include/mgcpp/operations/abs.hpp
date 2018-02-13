
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
}  // namespace strict
}  // namespace mgcpp

#include <mgcpp/operations/abs.tpp>
#endif
