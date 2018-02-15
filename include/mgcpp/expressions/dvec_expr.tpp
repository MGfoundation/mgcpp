
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dvec_expr.hpp>
//#include <mgcpp/expressions/forward.hpp>

namespace mgcpp {
template <typename DenseMatrix, typename Type, size_t DeviceId>
inline decltype(auto) eval(
    dense_vector<DenseMatrix, Type, DeviceId> const& device_vec) {
  return ~device_vec;
}
}  // namespace mgcpp
