
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DMAT_REF_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DMAT_REF_EXPR_HPP_

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/matrix/forward.hpp>
#include <mgcpp/expressions/eval_context.hpp>
#include <mgcpp/expressions/generic_op.hpp>

namespace mgcpp {

template <typename Matrix /* = device_matrix<float> */>
struct dmat_ref_expr : dmat_expr<dmat_ref_expr<Matrix>> {
  using value_type = typename Matrix::value_type;
  using result_type = Matrix;

  Matrix const& _mat;
  inline dmat_ref_expr(Matrix const& mat);

  inline void traverse(eval_context&) const {}
  inline Matrix const& eval(eval_context& ctx) const;
};

template <typename DenseMatrix, typename Type, size_t DeviceId>
inline dmat_ref_expr<DenseMatrix> ref(
    dense_matrix<DenseMatrix, Type, DeviceId> const& mat);
}  // namespace mgcpp

#endif
