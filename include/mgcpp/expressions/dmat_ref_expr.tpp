
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dmat_ref_expr.hpp>

namespace mgcpp {

template <typename DenseMatrix, typename Type>
inline dmat_ref_expr<DenseMatrix> ref(
    dense_matrix<DenseMatrix, Type> const& mat) {
  return dmat_ref_expr<DenseMatrix>(~mat);
}

}  // namespace mgcpp
