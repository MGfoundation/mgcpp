
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_RESULT_TYPE_HPP_
#define _MGCPP_EXPRESSIONS_RESULT_TYPE_HPP_

#include <type_traits>
#include <mgcpp/type_traits/type_traits.hpp>

namespace mgcpp
{
    template<typename T, typename = mgcpp::void_t<>>
    struct result_type {};
}

#endif
