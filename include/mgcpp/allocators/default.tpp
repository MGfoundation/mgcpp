
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/allocators/default.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/system/type_cast.hpp>

namespace mgcpp
{
    template<typename T, size_t DeviceId>
    typename default_allocator<T, DeviceId>::pointer
    default_allocator<T, DeviceId>::
    allocate(size_t n) 
    { return _alloc_tr::allocate(_alloc, n); }

    template<typename T, size_t DeviceId>
    void
    default_allocator<T, DeviceId>::
    deallocate(pointer p, size_t n)
    { return _alloc_tr::deallocate(_alloc, p, n); }

    template<typename T, size_t DeviceId>
    typename default_allocator<T, DeviceId>::device_pointer
    default_allocator<T, DeviceId>::
    device_allocate(size_t n) const
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error()); }

        auto ptr = cuda_malloc<device_value_type>(n);
        if(!ptr)
        { MGCPP_THROW_SYSTEM_ERROR(ptr.error()); }
        return ptr.value();
    }

    template<typename T, size_t DeviceId>
    void 
    default_allocator<T, DeviceId>::
    device_deallocate(device_pointer p, size_t n) const
    {
        (void)n;
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error()); }

        auto free_stat = cuda_free<device_value_type>(p);
        if(!p)
        { MGCPP_THROW_SYSTEM_ERROR(free_stat.error()); }
    }

    template<typename T, size_t DeviceId>
    template<typename>
    void
    default_allocator<T, DeviceId>::
    copy_from_host(device_pointer device, const_pointer host, size_t n) const
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error()); }

        outcome::result<void> cpy_stat = outcome::success();
        if (is_reinterpretable<T>::value)
        {
            cpy_stat = cuda_memcpy(reinterpret_cast<pointer>(device),
                                        host,
                                        n,
                                        cuda_memcpy_kind::host_to_device);
        }
        else
        {
            std::vector<device_value_type> host_d(n);
            mgcpp_cast(host, host + n, host_d.data());
            cpy_stat = cuda_memcpy(device,
                                        host_d.data(),
                                        n,
                                        cuda_memcpy_kind::host_to_device);
        }
        if(!cpy_stat)
        { MGCPP_THROW_SYSTEM_ERROR(cpy_stat.error()); }
    }

    template<typename T, size_t DeviceId>
    template<typename>
    void
    default_allocator<T, DeviceId>::
    copy_to_host(pointer host, const_device_pointer device, size_t n) const
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error()); }

        outcome::result<void> cpy_stat = outcome::success();
        if (is_reinterpretable<T>::value)
        {
            cpy_stat = cuda_memcpy(host,
                                        reinterpret_cast<const_pointer>(device),
                                        n,
                                        cuda_memcpy_kind::device_to_host);
        }
        else
        {
            std::vector<device_value_type> host_d(n);
            cpy_stat = cuda_memcpy(host_d.data(),
                                        device,
                                        n,
                                        cuda_memcpy_kind::device_to_host);
            mgcpp_cast(host_d.data(), host_d.data() + n, host);
        }
        if(!cpy_stat)
        { MGCPP_THROW_SYSTEM_ERROR(cpy_stat.error()); }
    }
}
