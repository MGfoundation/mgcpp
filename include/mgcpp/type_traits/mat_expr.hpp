
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_MAT_EXPR_HPP_
#define _MGCPP_TYPE_TRAITS_MAT_EXPR_HPP_

#include <type_traits>

#include <mgcpp/expressions/expr_result_type.hpp>
#include <mgcpp/type_traits/gpu_mat.hpp>

namespace mgcpp
{
    template<typename Head>
    struct assert_mat_expr
    {
        using result = typename std::enable_if<
            is_gpu_matrix<
                typename result_type<Head>::type>::value>::type;
    };

    template<typename Head1, typename Head2>
    struct assert_both_mat_expr
    {
        using result = typename std::enable_if<
            is_gpu_matrix<
                typename result_type<Head1>::type>::value
            &&  is_gpu_matrix<
                typename result_type<Head2>::type>::value
            >::type;
    };
}

#endif
