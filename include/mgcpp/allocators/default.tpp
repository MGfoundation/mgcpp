
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/allocators/default.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    template<typename T>
    T* 
    default_allocator::
    allocate(size_t n) const
    {
        auto ptr = cuda_malloc<T>(n);
        if(!ptr)
        {
            MGCPP_THROW_SYSTEM_ERROR(ptr.error());
        }
        return ptr.value();
    }

    template<typename T>
    T* 
    default_allocator::
    allocate(size_t n, size_t device_id) const
    {
        auto set_device_stat = cuda_set_device(device_id);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto ptr = cuda_malloc<T>(n);
        if(!ptr)
        {
            MGCPP_THROW_SYSTEM_ERROR(ptr.error());
        }
        return ptr.value();
    }
    
    template<typename T>
    void 
    default_allocator::
    deallocate(T* p) const
    {
        auto free_stat = cuda_free<T>(n);
        if(!ptr)
        {
            MGCPP_THROW_SYSTEM_ERROR(free_stat.error());
        }
    }

    template<typename T>
    void 
    default_allocator::
    deallocate(T* p, size_t device_id) const
    {
        auto set_device_stat = cuda_set_device(device_id);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto free_stat = cuda_free<T>(p);
        if(!ptr)
        {
            MGCPP_THROW_SYSTEM_ERROR(free_stat.error());
        }
    }

    template<typename T>
    void
    default_allocator::
    copy_from_host(T* device, T const* host, size_t n) const
    {
        
        auto cpy_stat =
            cuda_memcpy(device, host, n,
                        cuda_memcpy_kind::host_to_device);
        if(!cpy_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(cpy_stat.error());
        }
    }

    template<typename T>
    void
    default_allocator::
    copy_from_host(T* device, T const* host, size_t n,
                   size_t device_id) const
    {
        auto set_device_stat = cuda_set_device(device_id);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto cpy_stat =
            cuda_memcpy(device, host, n,
                        cuda_memcpy_kind::host_to_device);
        if(!cpy_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(cpy_stat.error());
        }
    }

    template<typename T>
    void
    default_allocator::
    copy_to_host(T* host, T const* device, size_t n) const
    {
        auto cpy_stat =
            cuda_memcpy(host, device, n,
                        cuda_memcpy_kind::device_to_host);
        if(!cpy_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(cpy_stat.error());
        }
    }

    template<typename T>
    void
    default_allocator::
    copy_to_host(T* host, T const* device, size_t n,
                 size_t device_id) const
    {
        auto set_device_stat = cuda_set_device(device_id);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto cpy_stat =
            cuda_memcpy(host, device, n,
                        cuda_memcpy_kind::device_to_host);
        if(!cpy_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(cpy_stat.error());
        }
    }
}
