
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DVEC_REF_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DVEC_REF_EXPR_HPP_

#include <mgcpp/expressions/dvec_expr.hpp>
#include <mgcpp/expressions/generic_expr.hpp>

namespace mgcpp {

struct dvec_ref_expr_type;

template <typename Vector>
using dvec_ref_expr =
    generic_expr<dvec_ref_expr_type, 0, dvec_expr, Vector, 1, Vector const&>;

template <typename DenseVector, typename Type, size_t DeviceId>
inline dvec_ref_expr<DenseVector> ref(
    dense_vector<DenseVector, Type, DeviceId> const& mat);
}  // namespace mgcpp

#include <mgcpp/expressions/dvec_ref_expr.tpp>
#endif
