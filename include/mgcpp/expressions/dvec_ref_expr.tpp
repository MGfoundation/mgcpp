
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dvec_ref_expr.hpp>

namespace mgcpp {

template <typename DenseVector, typename Type, size_t DeviceId>
inline dvec_ref_expr<DenseVector> ref(
    dense_vector<DenseVector, Type, DeviceId> const& vec) {
  return dvec_ref_expr<DenseVector>(~vec);
}
}  // namespace mgcpp
