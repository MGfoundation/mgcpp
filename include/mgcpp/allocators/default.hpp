
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_ALLOCATORS_DEFAULT_HPP_
#define _MGCPP_ALLOCATORS_DEFAULT_HPP_

#include <cstdlib>

namespace mgcpp
{
    template<typename T>
    class default_allocator
    {
        using value_type = T; 

        inline T* allocate(size_t n) const;
        inline T* allocate(size_t n,
                           size_t device_id) const;

        inline void deallocate(T* p) const;
        inline void deallocate(T* p,
                               size_t device_id) const;

        inline void copy_from_host(T* device, T const* host, size_t n) const;
        inline void copy_from_host(T* device, T const* host, size_t n,
                                   size_t device_id) const;

        inline void copy_to_host(T* host, T const* device, size_t n) const;
        inline void copy_to_host(T* host, T const* device, size_t n,
                                 size_t device_id) const;
    };
}

#include <mgcpp/allocators/default.tpp>
#endif
