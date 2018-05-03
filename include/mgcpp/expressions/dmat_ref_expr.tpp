
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dmat_ref_expr.hpp>

namespace mgcpp {

template <typename DenseMatrix, typename Type, size_t DeviceId>
inline dmat_ref_expr<DenseMatrix, Type, DeviceId>::dmat_ref_expr(
    DenseMatrix const& mat)
    : _mat(mat) {}

template <typename DenseMatrix, typename Type, size_t DeviceId>
inline DenseMatrix const& dmat_ref_expr<DenseMatrix, Type, DeviceId>::eval(eval_context&)
    const {
  return _mat;
}

template <typename DenseMatrix, typename Type, size_t DeviceId>
inline dmat_ref_expr<DenseMatrix, Type, DeviceId> ref(
    dense_matrix<DenseMatrix, Type, DeviceId> const& mat) {
  return dmat_ref_expr<DenseMatrix, Type, DeviceId>(~mat);
}

}  // namespace mgcpp
