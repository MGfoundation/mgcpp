
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_VECTOR_DENSE_VECTOR_HPP_
#define _MGCPP_VECTOR_DENSE_VECTOR_HPP_

#include <mgcpp/expressions/dvec_expr.hpp>
#include <mgcpp/vector/vector_base.hpp>

namespace mgcpp {
template <typename DenseVecType, typename Type>
class dense_vector : public vector_base<DenseVecType, Type> {};
}  // namespace mgcpp

#endif
