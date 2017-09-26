
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_GPU_VECTOR_HPP_
#define _MGCPP_GPU_VECTOR_HPP_

#include <mgcpp/cpu/fwd.hpp>

namespace mgcpp
{
    namespace gpu
    {
        enum class allignment
        {
            row, column
        };

        template<typename ElemType,
                 allignment Allign,
                 size_t DeviceId>
        class vector
        {
        private:
            ElemType* _data;
            size_t _id;

        public:
            inline vector();

            inline vector(size_t x_dim, size_t y_dim);

            inline vector(size_t x_dim, size_t y_dim, ElemType init);

            // template<size_t Xdim, size_t Ydim>
            // inline vector(dynamic_vector<ElemType> const& cpu_mat);

            // template<size_t Xdim, size_t Ydim>
            // inline vector(
            //     static_vector<ElemType, Xdim, Ydim> const& cpu_mat);

            // inline dynamic_vector<ElemType>
            // copy_to_cpu() const;

            // template<size_t Xdim, size_t Ydim>
            // inline static_vector<ElemType, Xdim, Ydim>
            // copy_to_cpu() const;

            inline ElemType const*
            get_data() const;

            inline ElemType*
            get_data_mutable();

            inline ~vector();
        }
    }
}

#endif
