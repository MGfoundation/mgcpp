
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_MAT_EXPR_HPP_
#define _MGCPP_TYPE_TRAITS_MAT_EXPR_HPP_

#include <type_traits>

#include <mgcpp/expressions/mat_expr.hpp>
#include <mgcpp/device/forward.hpp>

namespace mgcpp
{
    template<typename T>
    struct is_mat_expr_impl : std::false_type {};

    template<typename Expr>
    struct is_mat_expr_impl<mat_expr<Expr>> : std::true_type {};

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    struct is_mat_expr_impl<device_matrix<Type, DeviceId, Alloc>>
        : std::true_type {};

    template<typename DeviceMat>
    struct is_mat_expr
    {
        enum { value = is_mat_expr_impl<
               typename std::decay<DeviceMat>::type
               >::value };
    };
}

#endif
