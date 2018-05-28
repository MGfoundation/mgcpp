
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_MATRIX_FORWARD_HPP_
#define _MGCPP_MATRIX_FORWARD_HPP_

#include <cstddef>

namespace mgcpp {
template <typename MatrixType, typename Type>
class matrix;

template <typename DenseMatrixType, typename Type>
class dense_matrix;

template <typename Type, typename Alloc>
class device_matrix;
}  // namespace mgcpp

#endif
