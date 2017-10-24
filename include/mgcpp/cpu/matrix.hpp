
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CPU_MATRIX_HPP_
#define _MGCPP_CPU_MATRIX_HPP_

#include <mgcpp/gpu/forward.hpp>
#include <mgcpp/cpu/forward.hpp>
#include <mgcpp/global/storage_order.hpp>

#include <cstdlib>

namespace mgcpp
{
    namespace cpu
    {
        template<typename T,
                 storage_order SO>
        class matrix
        {
        private:
            T* _data; 
            size_t _m_dim;
            size_t _n_dim;
        
        public:
            inline matrix() noexcept;

            inline ~matrix() noexcept;

            inline matrix(size_t i, size_t j);

            inline matrix(size_t i, size_t j, T init);

            inline matrix(size_t i, size_t j,
                          T* data) noexcept;

            // template<size_t DeviceId>
            // inline matrix(
            //     gpu::matrix<T, DeviceId> const& gpu_mat);

            // template<size_t DeviceId>
            // inline gpu::matrix<T, DeviceId, >
            // copy_to_gpu() const;

            inline T
            operator()(size_t i, size_t j) const;

            inline T&
            operator()(size_t i, size_t j);

            inline T const*
            get_data() const;

            inline T*
            get_data_mutable() const;

            inline std::pair<size_t, size_t>
            shape() const noexcept;
        };
    }
}

#include <mgcpp/cpu/matrix.tpp>
#endif
