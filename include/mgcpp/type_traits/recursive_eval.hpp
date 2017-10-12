//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_RECURSIVE_EVAL_HPP_
#define _MGCPP_TYPE_TRAITS_RECURSIVE_EVAL_HPP_

#include <type_traits>

namespace mgcpp
{
    template<typename F,
             typename Head, typename... Tail>
    struct fold_and
    {
        enum { value = F<Head>::value && fold_and<F, Tail>::value };
    };
    
    template <typename F, typename Head>
    struct fold_and
    {
        enum { value  = F<Head>::value };
    };
    
    template<typename F,
             typename Head, typename... Tail>
    struct fold_or
    {
        enum { value = F<Head>::value || fold_and<F, Tail>::value };
    };
    
    template<typename F, typename Head>
    struct fold_or
    {
        enum { value = F<Head>::value };
    };
}
