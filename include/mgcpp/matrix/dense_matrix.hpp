
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_MATRIX_DENSE_MATRIX_HPP_
#define _MGCPP_MATRIX_DENSE_MATRIX_HPP_

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/matrix/matrix_base.hpp>

namespace mgcpp {
template <typename DenseMatType, typename Type, size_t DeviceId>
class dense_matrix : public matrix_base<DenseMatType>,
                     public dmat_expr<DenseMatType> {
  using result_type = DenseMatType;
};
}  // namespace mgcpp

#endif
