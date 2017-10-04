
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_GPU_MATRIX_HPP_
#define _MGCPP_GPU_MATRIX_HPP_

#include <mgcpp/cpu/forward.hpp>
#include <mgcpp/global/storage_order.hpp>
#include <mgcpp/context/thread_context.hpp>

namespace mgcpp
{
    namespace gpu
    {
        template<typename ElemType,
                 size_t DeviceId = 0,
                 storage_order SO = row_major>
        class matrix
        {
        private:
            ElemType* _data;
            thread_context* _context;
            size_t _m_dim;
            size_t _n_dim;
            bool _released;

        public:
            inline matrix() noexcept;

            inline ~matrix() noexcept;

            inline matrix(thread_context& context) noexcept;

            inline matrix(size_t i, size_t j);

            inline matrix(thread_context& context,
                          size_t i, size_t j);

            inline matrix(size_t i, size_t j, ElemType init);

            inline matrix(thread_context& context,
                          size_t i, size_t j, ElemType init);

            inline matrix(cpu::matrix<ElemType, SO> const& cpu_mat);

            inline matrix<ElemType, DeviceId, SO>&
            zeros();

            inline matrix<ElemType, DeviceId, SO>&
            resize(size_t i, size_t j);

            inline matrix<ElemType, DeviceId, SO>&
            resize(size_t i, size_t j, ElemType init);

            inline matrix<ElemType, DeviceId, SO>&
            copy_from_host(cpu::matrix<ElemType, SO> const& cpu_mat);

            inline cpu::matrix<ElemType, SO>
            copy_to_host() const;

            inline ElemType
            check_value(size_t i, size_t j) const;

            inline ElemType const*
            get_data() const;

            inline ElemType*
            get_data_mutable();

            inline thread_context*
            get_thread_context() const noexcept;

            inline ElemType*
            release_data();

            inline std::pair<size_t, size_t>
            shape() const noexcept;
        };
    }
}

#include <mgcpp/gpu/matrix.tpp>
#endif
