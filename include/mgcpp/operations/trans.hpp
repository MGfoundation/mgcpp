
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_TRANS_HPP_
#define _MGCPP_OPERATIONS_TRANS_HPP_

#include <mgcpp/matrix/forward.hpp>

namespace mgcpp {
namespace strict {
template <typename DenseMat, typename Type>
decltype(auto) trans(dense_matrix<DenseMat, Type> const& mat);
}
}  // namespace mgcpp

#include <mgcpp/operations/trans.tpp>
#endif
