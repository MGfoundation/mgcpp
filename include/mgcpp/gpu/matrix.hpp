
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_GPU_MATRIX_HPP_
#define _MGCPP_GPU_MATRIX_HPP_

#include <mgcpp/cpu/fwd.hpp>

namespace mg
{
    namespace gpu
    {
        template<typename ElemType, size_t DeviceId>
        class matrix
        {
        private:
            ElemType* _data;
            size_t _x_dim;
            size_t _y_dim;
            bool _released;

        public:
            inline matrix();

            inline matrix(size_t x_dim, size_t y_dim);

            inline matrix(size_t x_dim, size_t y_dim, ElemType init);

            inline matrix(size_t x_dim, size_t y_dim,
                          std::nothrow_t const& nothrow_flag);

            inline matrix(size_t x_dim, size_t y_dim, ElemType init
                          std::nothrow_t const& nothrow_flag);

            template<size_t Xdim, size_t Ydim>
            inline matrix(dynamic_matrix<ElemType> const& cpu_mat);

            template<size_t Xdim, size_t Ydim>
            inline matrix(
                static_matrix<ElemType, Xdim, Ydim> const& cpu_mat);

            inline cpu::matrix<ElemType>
            copy_to_cpu() const;
   
            inline ElemType const*
            get_data() const;

            inline ElemType*
            get_data_mutable();

            inline ElemType*
            release_data();

            inline ~matrix();
        };
    }
}

#endif
