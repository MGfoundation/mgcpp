
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_ALLOCATORS_DEFAULT_HPP_
#define _MGCPP_ALLOCATORS_DEFAULT_HPP_

#include <cstdlib>
#include <memory>
#include <mgcpp/type_traits/device_pointer_type.hpp>

namespace mgcpp
{
    template<typename T, size_t DeviceId>
    struct default_allocator : std::allocator<T>
    {
        typedef std::allocator<T> Alloc;
        typedef std::allocator_traits<Alloc> _alloc_tr;
        using device_pointer = typename device_pointer<T>::type;
        using const_device_pointer = typename const_device_pointer<T>::type;

        Alloc _alloc;

        inline T* allocate(size_t n);

        inline void deallocate(T* p, size_t n);

        inline device_pointer device_allocate(size_t n) const;

        inline void device_deallocate(device_pointer p, size_t n) const;

        inline void
        copy_from_host(device_pointer device, T const* host, size_t n) const;

        inline void
        copy_to_host(T* host, const_device_pointer device, size_t n) const;
    };
}

#include <mgcpp/allocators/default.tpp>
#endif
