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
