
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/matrix/matrix_base.hpp>

namespace mgcpp {
template <typename MatrixType, typename Type>
MatrixType const& matrix_base<MatrixType, Type>::operator~() const
    noexcept {
  return *static_cast<MatrixType const*>(this);
};

template <typename MatrixType, typename Type>
MatrixType& matrix_base<MatrixType, Type>::operator~() noexcept {
  return *static_cast<MatrixType*>(this);
};
}  // namespace mgcpp
