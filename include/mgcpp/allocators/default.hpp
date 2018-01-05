
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_ALLOCATORS_DEFAULT_HPP_
#define _MGCPP_ALLOCATORS_DEFAULT_HPP_

#include <cstdlib>
#include <memory>
#include <mgcpp/type_traits/device_value_type.hpp>
#include <mgcpp/type_traits/host_value_type.hpp>
#include <mgcpp/type_traits/type_traits.hpp>

namespace mgcpp
{
    template<typename Type, size_t DeviceId>
    struct default_allocator
    {
        using value_type = typename value_type<Type>::type;
        using pointer = value_type*;
        using const_pointer = value_type const*;
        using device_value_type = typename device_value_type<Type>::type;
        using device_pointer = device_value_type*;
        using const_device_pointer = device_value_type const*;

        template<typename NewType>
        using rebind_alloc = default_allocator<NewType, DeviceId>;

        typedef std::allocator<value_type> Alloc;
        typedef std::allocator_traits<Alloc> _alloc_tr;

        Alloc _alloc;

        inline pointer allocate(size_t n);

        inline void deallocate(pointer p, size_t n);

        inline device_pointer device_allocate(size_t n) const;

        inline void
        device_deallocate(device_pointer p, size_t n) const;

        template<typename = void>
        inline auto
        copy_from_host(device_pointer device, const_pointer host, size_t n) const
            -> std::enable_if_t<!is_reinterpretable<Type>::value>;

        template<typename = void>
        inline auto
        copy_from_host(device_pointer device, const_pointer host, size_t n) const
            -> std::enable_if_t<is_reinterpretable<Type>::value>;

        template<typename = void>
        inline auto
        copy_to_host(pointer host, const_device_pointer device, size_t n) const
            -> std::enable_if_t<!is_reinterpretable<Type>::value>;

        template<typename = void>
        inline auto
        copy_to_host(pointer host, const_device_pointer device, size_t n) const
            -> std::enable_if_t<is_reinterpretable<Type>::value>;
    };
}

#include <mgcpp/allocators/default.tpp>
#endif
