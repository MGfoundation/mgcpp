
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/vector/vector_base.hpp>

namespace mgcpp {
template <typename VectorType, typename Type>
VectorType const& vector_base<VectorType, Type>::operator~() const
    noexcept {
  return *static_cast<VectorType const*>(this);
};

template <typename VectorType, typename Type>
VectorType& vector_base<VectorType, Type>::operator~() noexcept {
  return *static_cast<VectorType*>(this);
};
}  // namespace mgcpp
