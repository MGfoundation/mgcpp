#ifndef _DYNAMIC_MATRIX_HPP_
#define _DYNAMIC_MATRIX_HPP_

#include <mgcpp/utility/declarations.hpp>

namespace mg
{
    template<typename ElemType>
    class dynamic_matrix
    {
    private:
        ElemType* _data; 
        
    public:
        inline dynamic_matrix();

        inline dynamic_matrix(size_t x_dim, size_t y_dim);

        inline dynamic_matrix(size_t x_dim, size_t y_dim,
                              ElemType init);

        template<size_t DeviceId>
        inline dynamic_matrix(
            gpu_matrix<ElemType, DeviceId> const& gpu_mat);

        template<size_t DeviceId>
        inline gpu_matrix<ElemType, DeviceId>
        copy_to_gpu() const;
    };
}

#endif
