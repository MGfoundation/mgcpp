
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_VECTOR_VECTOR_BASE_HPP_
#define _MGCPP_VECTOR_VECTOR_BASE_HPP_

#include <cstdlib>

namespace mgcpp {
template <typename VectorType, typename Type, size_t DeviceId>
struct vector_base {};
}  // namespace mgcpp

#include <mgcpp/vector/vector_base.tpp>
#endif
