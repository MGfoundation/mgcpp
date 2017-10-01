
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_GPU_MATRIX_HPP_
#define _MGCPP_GPU_MATRIX_HPP_

#include <mgcpp/cpu/forward.hpp>
#include <mgcpp/context/thread_context.hpp>

namespace mgcpp
{
    enum class storage_order { column_major = 0, row_major}; 

    namespace gpu
    {
        template<typename ElemType,
                 size_t DeviceId = 0,
                 storage_order StoreOrder = storage_order::row_major>
        class matrix
        {
        private:
            ElemType* _data;
            thread_context* _context;
            size_t _row_dim;
            size_t _col_dim;
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

            inline matrix(cpu::matrix<ElemType> const& cpu_mat);

            inline matrix<ElemType, DeviceId, StoreOrder>&
            zeros();

            inline matrix<ElemType, DeviceId, StoreOrder>&
            resize(size_t i, size_t j);

            inline matrix<ElemType, DeviceId, StoreOrder>&
            resize(size_t i, size_t j, ElemType init);

            inline cpu::matrix<ElemType>
            copy_to_cpu() const;

            inline ElemType
            check_value(size_t i, size_t j) const;

            // inline ElemType const*
            // get_data() const;

            // inline ElemType*
            // get_data_mutable();

            // inline ElemType*
            // release_data();

            inline size_t
            rows() const noexcept;

            inline size_t
            columns() const noexcept;
        };
    }
}

#include <mgcpp/gpu/matrix.tpp>

#endif
