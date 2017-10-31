
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
    struct is_mat_expr : std::false_type {};

    template<typename Expr>
    struct is_mat_expr<mat_expr<Expr>> : std::true_type {};

    template<typename Type,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    struct is_mat_expr<device_matrix<Type, DeviceId, SO, Alloc>>
        : std::true_type {};

    template<typename Head>
    struct assert_mat_expr
    {
        using result = typename std::enable_if<
            is_mat_expr<
                typename std::decay<Head>::type>::value>::type;
    };

    template<typename First, typename Second>
    struct assert_both_mat_expr
    {
        using result =
            typename std::enable_if<
            is_mat_expr<
                typename std::decay<First>::type>::value
            && is_mat_expr<
                typename std::decay<Second>::type>::value>::type;
    };
}

#endif
