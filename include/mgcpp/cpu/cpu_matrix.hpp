
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CPU_MATRIX_HPP_
#define _MGCPP_CPU_MATRIX_HPP_

#include <mgcpp/gpu/fwd.hpp>

namespace mg
{
    namespace cpu
    {
        template<typename ElemType>
        class matrix
        {
        private:
            ElemType* _data; 
        
        public:
            inline matrix();

            inline matrix(size_t x_dim, size_t y_dim);

            inline matrix(size_t x_dim, size_t y_dim,
                          ElemType init);

            template<size_t DeviceId>
            inline matrix(gpu::matrix<ElemType, DeviceId> const& gpu_mat);

            template<size_t DeviceId>
            inline gpu::matrix<ElemType, DeviceId>
            copy_to_gpu() const;
        };
    }
}

#endif
