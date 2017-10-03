
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CPU_MATRIX_HPP_
#define _MGCPP_CPU_MATRIX_HPP_

#include <mgcpp/gpu/forward.hpp>
#include <mgcpp/global/storage_order.hpp>

#include <cstdlib>

namespace mgcpp
{
    namespace cpu
    {
        template<typename ElemType,
                 storage_order StoreOrder = row_major>
        class matrix
        {
        private:
            ElemType* _data; 
            size_t _row_dim;
            size_t _col_dim;
        
        public:
            inline matrix() noexcept;

            inline ~matrix() noexcept;

            inline matrix(size_t i, size_t j);

            inline matrix(size_t i, size_t j, ElemType init);

            // template<size_t DeviceId>
            // inline matrix(
            //     gpu::matrix<ElemType, DeviceId> const& gpu_mat);

            // template<size_t DeviceId>
            // inline gpu::matrix<ElemType, DeviceId, >
            // copy_to_gpu() const;

            inline ElemType
            operator()(size_t i, size_t j) const;

            inline ElemType&
            operator()(size_t i, size_t j);

            inline ElemType const*
            get_data() const;

            inline ElemType*
            get_data_mutable() const;

            inline size_t
            rows() const noexcept;

            inline size_t
            columns() const noexcept;
        };
    }
}

#include <mgcpp/cpu/matrix.tpp>
#endif
