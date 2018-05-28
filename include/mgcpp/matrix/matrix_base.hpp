
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_MATRIX_MATRIX_BASE_HPP_
#define _MGCPP_MATRIX_MATRIX_BASE_HPP_

#include <cstddef>

namespace mgcpp {
template <typename MatrixType, typename Type>
struct matrix_base {
  inline MatrixType const& operator~() const noexcept;

  inline MatrixType& operator~() noexcept;
};
}  // namespace mgcpp

#include <mgcpp/matrix/matrix_base.tpp>
#endif
