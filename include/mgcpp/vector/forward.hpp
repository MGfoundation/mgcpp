
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_VECTOR_FORWARD_HPP_
#define _MGCPP_VECTOR_FORWARD_HPP_

#include <cstddef>

namespace mgcpp {
template <typename VectorType, typename Type>
struct vector_base;

template <typename DenseVecType, typename Type>
class dense_vector;

template <typename Type, typename Alloc>
class device_vector;
}  // namespace mgcpp

#endif
