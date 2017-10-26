
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_GPU_VECTOR_HPP_
#define _MGCPP_GPU_VECTOR_HPP_

#include <cstdlib>

#include <mgcpp/cpu/forward.hpp>
#include <mgcpp/device/forward.hpp>
#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/global/allignment.hpp>

namespace mgcpp
{
    template<typename T,
             size_t DeviceId,
             allignment Allign>
    class device_vector
    {
    private:
        T* _data;
        thread_context* _context;
        size_t _size;
        bool _released;

    public:
        inline device_vector() noexcept;

        inline ~device_vector() noexcept;

        inline device_vector(size_t size);

        inline device_vector(size_t size, T init);

        inline
        device_vector(device_vector<T, DeviceId, Allign> const& other);

        inline
        device_vector(device_vector<T, DeviceId, Allign>&& other) noexcept;

        inline device_vector<T, DeviceId, Allign>&
        operator=(device_vector<T, DeviceId, Allign> const& other);

        inline device_vector<T, DeviceId, Allign>&
        operator=(device_vector<T, DeviceId, Allign>&& other) noexcept;

        inline device_vector<T, DeviceId, Allign>&
        zero();

        inline void 
        copy_from_host(cpu::vector<T, Allign> const& host);

        inline cpu::vector<T, Allign>
        copy_to_host() const;

        inline T
        check_value(size_t i) const;

        inline T const*
        data() const noexcept;

        inline T*
        data_mutable() noexcept;

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
