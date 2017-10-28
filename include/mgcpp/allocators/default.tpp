
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
    template<typename T, size_t DeviceId>
    T*
    default_allocator<T, DeviceId>::
    allocator(size_t n) const
    { return _alloc_tr::allocate(_alloc, n); }

    template<typename T, size_t DeviceId>
    void
    default_allocator<T, DeviceId>::
    allocator(T* p, size_t n) const
    { return _alloc_tr::deallocate(_alloc, p, n); }

    template<typename T, size_t DeviceId>
    T* 
    default_allocator<T, DeviceId>::
    device_allocate(size_t n) const
    {
        auto set_device_stat = cuda_set_device(DeviceId);
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

    template<typename T, size_t DeviceId>
    void 
    default_allocator<T, DeviceId>::
    device_deallocate(T* p, size_t n) const
    {
        (void)n;
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto free_stat = cuda_free<T>(p);
        if(!p)
        {
            MGCPP_THROW_SYSTEM_ERROR(free_stat.error());
        }
    }

    template<typename T, size_t DeviceId>
    void
    default_allocator<T, DeviceId>::
    copy_from_host(T* device, T const* host, size_t n) const
    {
        auto set_device_stat = cuda_set_device(DeviceId);
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

    template<typename T, size_t DeviceId>
    void
    default_allocator<T, DeviceId>::
    copy_to_host(T* host, T const* device, size_t n) const
    {
        auto set_device_stat = cuda_set_device(DeviceId);
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
