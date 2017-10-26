
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_ADDITION_HPP_
#define _MGCPP_OPERATIONS_ADDITION_HPP_

#include <mgcpp/device/forward.hpp>

namespace mgcpp
{
    namespace strict
    {
        template<typename T, size_t Device, storage_order SO>
        inline device_matrix<T, Device, SO>
        add(device_matrix<T, Device, SO> const& first,
            device_matrix<T, Device, SO> const& second);

        template<typename T, size_t Device, allignment Allign>
        inline device_vector<T, Device, Allign>
        add(device_vector<T, Device, Allign> const& first,
            device_vector<T, Device, Allign> const& second);
    }
}

#include <mgcpp/operations/add.tpp>
#endif
