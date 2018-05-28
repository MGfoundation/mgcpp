
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_PAD_HPP_
#define _MGCPP_OPERATIONS_PAD_HPP_

#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/type_traits/host_value_type.hpp>
#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

#include <tuple>

namespace mgcpp {
/// pad_size_t::first : pad before amount
/// pad_size_t::second : pad after amount
using pad_size_t = std::pair<size_t, size_t>;

namespace strict {
/**
 * Applies padding to a vector.
 * \param vec the vector to be padded.
 * \param pad the padding size.
 * \param pad_constant the value of the padding cells. If not specified, the
 * default padding is zero. \return the padded result
 */
template <typename DenseVec, typename Type>
inline decltype(auto) pad(dense_vector<DenseVec, Type> const& vec,
                          pad_size_t pad,
                          typename value_type<Type>::type pad_constant =
                              typename value_type<Type>::type{});
}  // namespace strict
}  // namespace mgcpp

#include <mgcpp/operations/pad.tpp>
#endif
