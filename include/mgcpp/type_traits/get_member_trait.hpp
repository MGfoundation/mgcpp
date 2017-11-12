
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_GET_MEMBER_TRAIT_HPP_
#define _MGCPP_TYPE_TRAITS_GET_MEMBER_TRAIT_HPP_

#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/vector/device_vector.hpp>

#include <type_traits>
#include <cstdlib>

namespace mgcpp
{
    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    class device_matrix;

    template<typename Type,
             allignment Allign,
             size_t DeviceId,
             typename Alloc>
    class device_vector;

    template<typename T>
    struct get_allocator_impl { };

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    struct get_allocator_impl<device_matrix<Type, DeviceId, Alloc>>
    { using type = Alloc; };

    template<typename Type,
             allignment Allign,
             size_t DeviceId,
             typename Alloc>
    struct get_allocator_impl<device_vector<Type, Allign, DeviceId, Alloc>>
    { using type = Alloc; };

    template<typename Type>
    struct get_allocator
    {
        using type =
            typename get_allocator_impl<
            typename std::decay<Type>::type>::type;
    };
}

#endif
