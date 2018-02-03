
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_PUN_CAST_HPP_
#define _MGCPP_PUN_CAST_HPP_

#include <type_traits>

namespace mgcpp
{
    template<typename To, typename From>
    inline To pun_cast(From const* from)
    {
        union pun_t { From src; typename std::remove_pointer<To>::type dst; };
        return &(reinterpret_cast<pun_t const*>(from))->dst;
    }

    template<typename To, typename From>
    inline To pun_cast(From* from)
    {
        union pun_t { From src; typename std::remove_pointer<To>::type dst; };
        return &(reinterpret_cast<pun_t*>(from))->dst;
    }
}

#endif
