
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_HADAMARD_HPP_
#define _MGCPP_OPERATIONS_HADAMARD_HPP_

#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

#include <cstdlib>

namespace mgcpp {
namespace strict {
/**
 * Element-wise multiplication of two equal-sized vectors. (Hadamard Product)
 * \param lhs left-hand side
 * \param rhs right-hand side
 * \returns the element-wise multiplication of lhs and rhs
 */
template <typename LhsDenseVec,
          typename RhsDenseVec,
          typename Type,
          size_t DeviceId>
inline decltype(auto) hdmd(
    dense_vector<LhsDenseVec, Type, DeviceId> const& lhs,
    dense_vector<RhsDenseVec, Type, DeviceId> const& rhs);
}  // namespace strict
}  // namespace mgcpp

#include <mgcpp/operations/hdmd.tpp>
#endif
