
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

namespace mgcpp
{
    template<typename T,
             storage_order SO>
    class host_matrix
    {
    private:
        T* _data; 
        size_t _m_dim;
        size_t _n_dim;
        
    public:
        inline host_matrix() noexcept;

        inline ~host_matrix() noexcept;

        inline host_matrix(size_t i, size_t j);

        inline host_matrix(size_t i, size_t j, T init);

        inline host_matrix(size_t i, size_t j,
                           T* data) noexcept;

        // template<size_t DeviceId>
        // inline matrix(
        //     gpu::matrix<T, DeviceId> const& gpu_mat);

        // template<size_t DeviceId>
        // inline gpu::matrix<T, DeviceId, >
        // copy_to_gpu() const;

        inline T
        operator()(size_t i, size_t j) const noexcept;

        inline T&
        operator()(size_t i, size_t j) noexcept;

        inline T
        at(size_t i, size_t j) const;

        inline T&
        at(size_t i, size_t j);

        inline T const*
        data() const;

        inline T*
        data_mutable() const;

        inline std::pair<size_t, size_t>
        shape() const noexcept;
    };
}

#include <mgcpp/host/matrix.tpp>
#endif
