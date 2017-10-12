
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_MAT_EXPR_HPP_
#define _MGCPP_TYPE_TRAITS_MAT_EXPR_HPP_

#include <type_traits>

#include <mgcpp/expressions/mat_mat_mult_expr.hpp>

namespace mgcpp
{
    template<T>
    struct is_gpu_expr : std::false_type {};

    template<typename GpuMat>
    struct is_gpu_expr<mat_mat_mult_expr<GpuMat>> : std::true_type {};
}

#endif
