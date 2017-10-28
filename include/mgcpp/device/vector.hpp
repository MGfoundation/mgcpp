
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_GPU_VECTOR_HPP_
#define _MGCPP_GPU_VECTOR_HPP_

#include <mgcpp/allocators/default.hpp>
#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/device/forward.hpp>
#include <mgcpp/global/allignment.hpp>
//#include <mgcpp/host/forward.hpp>

#include <cstdlib>
#include <initializer_list>

namespace mgcpp
{
    template<typename T,
             size_t DeviceId = 0,
             allignment Allign = row,
             typename Alloc = mgcpp::default_allocator<T, DeviceId>>
    class device_vector : public Alloc
    {
        using Alloc::allocate;
        using Alloc::deallocate;
        using Alloc::device_allocate;
        using Alloc::device_deallocate;
        using Alloc::copy_to_host;
        using Alloc::copy_from_host;

    private:
        thread_context* _context;
        size_t _shape;
        T* _data;
        size_t _capacity;

    public:
        inline device_vector() noexcept;

        inline ~device_vector() noexcept;

        inline device_vector(size_t size);

        inline device_vector(size_t size, T init);

        inline device_vector(size_t size, T const* data);

        inline
        device_vector(device_vector<T, DeviceId, Allign, Alloc> const& other);

        inline
        device_vector(device_vector<T, DeviceId, Allign, Alloc>&& other) noexcept;

        inline
        device_vector(std::initializer_list<T> const& array);

        // inline
        // device_vector(host_vector<T, Allign> const& other);

        inline device_vector<T, DeviceId, Allign, Alloc>&
        operator=(device_vector<T, DeviceId, Allign, Alloc> const& other);

        inline device_vector<T, DeviceId, Allign, Alloc>&
        operator=(device_vector<T, DeviceId, Allign, Alloc>&& other) noexcept;

        // inline device_vector<T, DeviceId, Allign>&
        // operator=(host_vector<T, Allign> const& other); 

        inline device_vector<T, DeviceId, Allign, Alloc>&
        zero();

        // inline host_vector<T, Allign>
        // copy_to_host() const;

        inline T
        check_value(size_t i) const;

        inline T const*
        data() const noexcept;

        inline T*
        data_mutable() noexcept;

        inline size_t
        capacity() const noexcept;

        inline T*
        release_data() noexcept;

        inline thread_context*
        context() const noexcept;

        inline size_t
        shape() const noexcept;

        inline size_t
        size() const noexcept;
    };
}

#include <mgcpp/device/vector.tpp>
#endif
