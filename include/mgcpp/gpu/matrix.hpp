
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_GPU_MATRIX_HPP_
#define _MGCPP_GPU_MATRIX_HPP_

#include <mgcpp/cpu/forward.hpp>
#include <mgcpp/gpu/forward.hpp>
#include <mgcpp/global/storage_order.hpp>
#include <mgcpp/context/global_context.hpp>
#include <mgcpp/context/thread_context.hpp>

namespace mgcpp
{
    namespace gpu
    {
        template<typename T,
                 size_t DeviceId,
                 storage_order SO>
        class matrix
        {
        private:
            T* _data;
            thread_context* _context;
            size_t _m_dim;
            size_t _n_dim;
            bool _released;

        public:
            inline matrix() noexcept;

            inline ~matrix() noexcept;

            inline matrix(size_t i, size_t j);

            inline matrix(size_t i, size_t j, T init);

            inline matrix(size_t i, size_t j,
                          T* init,
                          storage_order data_SO = SO); //need implementation

            inline matrix(size_t i, size_t j,
                          T** init,
                          storage_order data_SO = SO); //need implementation

            template<typename U, size_t Rows, size_t Cols>
            inline matrix(U  const array[Rows][Cols]); //need implementation

            inline matrix(cpu::matrix<T, SO> const& cpu_mat);

            inline
            matrix(gpu::matrix<T, DeviceId, SO> const& other);

            inline
            matrix(gpu::matrix<T, DeviceId, SO>&& other) noexcept;

            gpu::matrix<T, DeviceId, SO>&
            operator=(gpu::matrix<T, DeviceId, SO> const& other);

            gpu::matrix<T, DeviceId, SO>&
            operator=(gpu::matrix<T, DeviceId, SO>&& other) noexcept;

            inline matrix<T, DeviceId, SO>&
            zeros();

            inline matrix<T, DeviceId, SO>&
            resize(size_t i, size_t j);

            inline matrix<T, DeviceId, SO>&
            resize(size_t i, size_t j, T init);

            inline matrix<T, DeviceId, SO>&
            copy_from_host(cpu::matrix<T, SO> const& cpu_mat);

            inline cpu::matrix<T, SO>
            copy_to_host() const;

            inline T
            check_value(size_t i, size_t j) const;

            inline T const*
            get_data() const;

            inline T*
            get_data_mutable();

            inline thread_context*
            get_thread_context() const noexcept;

            inline T*
            release_data();

            inline std::pair<size_t, size_t>
            shape() const noexcept;
        };
    }
}

#include <mgcpp/gpu/matrix.tpp>
#endif
