
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DENSE_MAT_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DENSE_MAT_EXPR_HPP_

#include <mgcpp/expressions/expression.hpp>
#include <mgcpp/matrix/forward.hpp>
#include <mgcpp/system/concept.hpp>
#include <mgcpp/type_traits/type_traits.hpp>
//#include <mgcpp/expressions/forward.hpp>

#include <cstdlib>
#include <utility>

namespace mgcpp {
template <typename Expr>
struct dmat_expr : public expression<Expr> {};

template <typename DenseMat, typename Type, size_t DeviceId>
inline decltype(auto) eval(
    dense_matrix<DenseMat, Type, DeviceId> const& device_mat,
    bool eval_trans = true);

template <typename Type, size_t DeviceId, typename Alloc>
inline device_matrix<Type, DeviceId, Alloc> const& eval(
    device_matrix<Type, DeviceId, Alloc> const& device_mat,
    bool eval_trans = true);
}  // namespace mgcpp

#include <mgcpp/expressions/dmat_expr.tpp>
#endif
