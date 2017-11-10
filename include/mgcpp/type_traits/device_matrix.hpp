
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_DEVICE_MATRIX_HPP_
#define _MGCPP_TYPE_TRAITS_DEVICE_MATRIX_HPP_

#include <type_traits>

#include <mgcpp/device/forward.hpp>

namespace mgcpp
{
    template<typename Type>
    struct is_device_matrix_impl : std::false_type {};

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    struct is_device_matrix_impl<mgcpp::device_matrix<Type, DeviceId, Alloc>>
        : std::true_type {};

    template<typename DeviceMat>
    struct is_device_matrix
    {
        enum { value = is_device_matrix_impl<
               typename std::decay<DeviceMat>::type
               >::value };
    };
}

#endif
