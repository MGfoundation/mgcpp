
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_ADAPTERS_ADAPTER_BASE_HPP_
#define _MGCPP_ADAPTERS_ADAPTER_BASE_HPP_

#include <type_traits>

namespace mgcpp
{
    template<typename T>
    struct adapter : std::false_type {};
}

#endif
