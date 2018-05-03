
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DMAT_REF_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DMAT_REF_EXPR_HPP_

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/matrix/forward.hpp>
#include <mgcpp/expressions/eval_context.hpp>

namespace mgcpp {

template <typename DenseMatrix, typename Type, size_t DeviceId>
struct dmat_ref_expr : dmat_expr<dmat_ref_expr<DenseMatrix, Type, DeviceId>> {
  using value_type = Type;
  using result_type = DenseMatrix;

  DenseMatrix const& _mat;
  inline dmat_ref_expr(DenseMatrix const& mat);
  inline DenseMatrix const& eval(eval_context& ctx) const;
};

template <typename DenseMatrix, typename Type, size_t DeviceId>
inline dmat_ref_expr<DenseMatrix, Type, DeviceId> ref(
    dense_matrix<DenseMatrix, Type, DeviceId> const& mat);
}  // namespace mgcpp

#endif
