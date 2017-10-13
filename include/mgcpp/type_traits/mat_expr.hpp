
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_MAT_EXPR_HPP_
#define _MGCPP_TYPE_TRAITS_MAT_EXPR_HPP_

#include <type_traits>

#include <mgcpp/type_traits/gpu_mat.hpp>

namespace mgcpp
{
    template<typename T>
    struct assert_mat_expr 
    {
        using result = typename
            std::enable_if<is_gpu_matrix<T::result_type>>::type;
    };
}

#endif
