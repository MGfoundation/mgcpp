#ifndef _MGCPP_EXPRESSIONS_DMAT_REF_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DMAT_REF_EXPR_HPP_

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/matrix/dense_matrix.hpp>

namespace mgcpp {

template <typename DenseMatrix, typename Type, size_t DeviceId>
struct dmat_ref_expr : dmat_expr<dmat_ref_expr<DenseMatrix, Type, DeviceId>> {
  DenseMatrix const& _mat;
  inline dmat_ref_expr(DenseMatrix const& mat);
  inline DenseMatrix const& eval() const;
};

template <typename DenseMatrix, typename Type, size_t DeviceId>
inline decltype(auto) eval(
    dmat_ref_expr<DenseMatrix, Type, DeviceId> const& mat);

template <typename DenseMatrix, typename Type, size_t DeviceId>
inline dmat_ref_expr<DenseMatrix, Type, DeviceId> ref(
    dense_matrix<DenseMatrix, Type, DeviceId> const& mat);
}

#endif
