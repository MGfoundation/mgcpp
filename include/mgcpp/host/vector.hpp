
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CPU_VECTOR_HPP_
#define _MGCPP_CPU_VECTOR_HPP_

#include <cstdlib>
#include <initializer_list>

#include <mgcpp/host/forward.hpp>
#include <mgcpp/device/forward.hpp>
#include <mgcpp/global/allignment.hpp>
#include <mgcpp/context/thread_context.hpp>

namespace mgcpp
{
    template<typename T,
             allignment Allign>
    class host_vector
    {
    private:
        T* _data; 
        size_t _size;
        bool _released;
        
    public:
        inline host_vector() noexcept;

        inline ~host_vector() noexcept;

        inline host_vector(size_t size);

        inline host_vector(size_t size, T init);

        inline host_vector(size_t size, T* data) noexcept;

        inline host_vector(std::initializer_list<T> const& array) noexcept;

        inline host_vector(host_vector<T, Allign> const& other);

        inline host_vector(host_vector<T, Allign>&& other) noexcept;

        template<size_t DeviceId>
        inline
        host_vector(device_vector<T, DeviceId, Allign> const& gpu_mat);

        inline host_vector<T, Allign>&
        operator=(host_vector<T, Allign> const& other);

        inline host_vector<T, Allign>&
        operator=(host_vector<T, Allign>&& other) noexcept;

        template<size_t DeviceId>
        inline host_vector<T, Allign>&
        operator=(device_vector<T, DeviceId, Allign> const& gpu_mat);

        template<size_t DeviceId>
        inline device_vector<T, DeviceId, Allign>
        copy_to_gpu() const;

        inline T
        operator[](size_t i) const noexcept;

        inline T&
        operator[](size_t i) noexcept;

    inline T
    at(size_t i) const;

    inline T&
    at(size_t i);

    inline T const*
    data() const;

    inline T*
    released_data();

    inline T*
    data_mutable() noexcept;

    inline size_t
    shape() const noexcept;

    inline size_t
    size() const noexcept;
    };
}

#include <mgcpp/host/vector.tpp>
#endif
