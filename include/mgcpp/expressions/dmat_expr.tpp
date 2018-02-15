
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/forward.hpp>

namespace mgcpp {
template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc> const& eval(
    device_matrix<Type, DeviceId, Alloc> const& device_mat,
    bool eval_trans) {
  (void)eval_trans;
  return device_mat;
}
}  // namespace mgcpp
