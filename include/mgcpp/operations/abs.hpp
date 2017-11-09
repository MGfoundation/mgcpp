
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_ABSOLUTE_HPP_
#define _MGCPP_OPERATIONS_ABSOLUTE_HPP_

#include <mgcpp/device/forward.hpp>

namespace mgcpp
{
    namespace strict
    {
        template<typename T,
                 size_t Device,
                 allignment Allign,
                 typename Alloc>
        inline device_vector<T, Device, Allign, Alloc>
        abs(device_vector<T, Device, Allign, Alloc> const& vec);
    }
}

#include <mgcpp/operations/abs.tpp>
#endif
