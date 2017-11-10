
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_SUBSTRACTION_HPP_
#define _MGCPP_OPERATIONS_SUBSTRACTION_HPP_

#include <mgcpp/device/forward.hpp>

namespace mgcpp
{
    namespace strict
    {
        template<typename T, size_t Device, allignment Allign, typename Alloc>
        inline device_vector<T, Device, Allign, Alloc>
        sub(device_vector<T, Device, Allign, Alloc> const& first,
            device_vector<T, Device, Allign, Alloc> const& second);

        template<typename T, size_t Device, storage_order SO, typename Alloc>
        inline device_matrix<T, Device, SO, Alloc>
        sub(device_matrix<T, Device, SO, Alloc> const& first,
            device_matrix<T, Device, SO, Alloc> const& second);
    }
}

#include <mgcpp/operations/sub.tpp>
#endif
