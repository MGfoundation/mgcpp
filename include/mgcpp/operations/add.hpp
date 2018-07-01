
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_ADDITION_HPP_
#define _MGCPP_OPERATIONS_ADDITION_HPP_

#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

#include <cstdlib>

namespace mgcpp {
namespace strict {
/** Adds two same-sized matrices together.
 * \param lhs the left-hand side
 * \param rhs the right-hand side
 */
template <typename LhsDenseMat, typename RhsDenseMat, typename Type>
inline decltype(auto) add(dense_matrix<LhsDenseMat, Type> const& lhs,
                          dense_matrix<RhsDenseMat, Type> const& rhs);

/** Adds two same-sized vectors together.
 * \param first the left-hand side
 * \param second the right-hand side
 */
template <typename LhsDenseVec, typename RhsDenseVec, typename Type>
inline decltype(auto) add(dense_vector<LhsDenseVec, Type> const& first,
                          dense_vector<RhsDenseVec, Type> const& second);
}  // namespace strict
}  // namespace mgcpp

#include <mgcpp/operations/add.tpp>
#endif
