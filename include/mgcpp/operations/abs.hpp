
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_ABSOLUTE_HPP_
#define _MGCPP_OPERATIONS_ABSOLUTE_HPP_

#include <mgcpp/device/forward.hpp>
#include <mgcpp/system/concept.hpp>
#include <mgcpp/type_traits/device_matrix.hpp>

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

        // template<typename T, size_t Device, typename Alloc,
        //          template<typename, size_t, typename> class DeviceMatrix,
        //          MGCPP_CONCEPT(is_device_matrix<DeviceMatrix>::value)>
        // inline device_matrix<T, Device, Alloc>
        // abs(DeviceMatrix<T, Device, Alloc> const& mat);

        template<typename DeviceMatrix,
                 MGCPP_CONCEPT(is_device_matrix<DeviceMatrix>::value)>
        inline device_matrix<typename DeviceMatrix::value_type,
                             DeviceMatrix::device_id,
                             typename DeviceMatrix::allocator_type>
        abs(DeviceMatrix const& mat);
    }
}

#include <mgcpp/operations/abs.tpp>
#endif
