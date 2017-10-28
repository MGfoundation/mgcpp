
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CPU_MATRIX_HPP_
#define _MGCPP_CPU_MATRIX_HPP_

#include <mgcpp/device/forward.hpp>
#include <mgcpp/host/forward.hpp>
#include <mgcpp/global/storage_order.hpp>

#include <cstdlib>
#include <initializer_list>

namespace mgcpp
{
    template<typename T,
             storage_order SO>
    class host_matrix
    {
    private:
        T* _data; 
        bool _released;
        size_t _m_dim;
        size_t _n_dim;
        
    public:
        inline host_matrix() noexcept;

        inline ~host_matrix() noexcept;

        inline host_matrix(size_t i, size_t j);

        inline host_matrix(size_t i, size_t j, T init);

        inline host_matrix(size_t i, size_t j, T* data) noexcept;

        inline
        host_matrix(host_matrix<T, SO> const& gpu_mat);

        inline
        host_matrix(host_matrix<T, SO>&& gpu_mat) noexcept;

        inline
        host_matrix(std::initializer_list<
                    std::initializer_list<T>> const& gpu_mat) noexcept;

        template<size_t DeviceId>
        inline
        host_matrix(device_matrix<T, DeviceId, SO> const& gpu_mat);

        inline host_matrix<T, SO>&
        operator=(host_matrix<T, SO> const& other);

        inline host_matrix<T, SO>&
        operator=(host_matrix<T, SO>&& other) noexcept;

        template<size_t DeviceId>
        inline host_matrix<T, SO>&
        operator=(device_matrix<T, DeviceId, SO> const& gpu_mat);

        template<size_t DeviceId> 
        inline device_matrix<T, DeviceId, SO>
        copy_to_device() const;

        inline T
        operator()(size_t i, size_t j) const noexcept;

        inline T&
        operator()(size_t i, size_t j) noexcept;

        inline T
        at(size_t i, size_t j) const;

        inline T&
        at(size_t i, size_t j);

        inline T const*
        data() const noexcept;

        inline T*
        data_mutable() noexcept;

        inline T*
        release_data() noexcept;

        inline std::pair<size_t, size_t>
        shape() const noexcept;
    };
}

#include <mgcpp/host/matrix.tpp>
#endif
