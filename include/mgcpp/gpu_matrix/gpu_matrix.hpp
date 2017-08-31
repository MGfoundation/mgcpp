#ifndef _GPU_MATRIX_HPP_
#define _GPU_MATRIX_HPP_

#include <mgcpp/utility/declarations.hpp>

namespace mg
{
    template<typename ElemType, size_t DeviceId>
    class gpu_matrix
    {
    private:
        ElemType* _data;
        size_t _id;

    public:
        inline gpu_matrix();

        inline gpu_matrix(size_t x_dim, size_t y_dim);

        inline gpu_matrix(size_t x_dim, size_t y_dim, ElemType init);

        template<size_t Xdim, size_t Ydim>
        inline gpu_matrix(
            dynamic_matrix<ElemType> const& cpu_mat);

        template<size_t Xdim, size_t Ydim>
        inline gpu_matrix(
            static_matrix<ElemType, Xdim, Ydim> const& cpu_mat);

        inline dynamic_matrix<ElemType>
        copy_to_cpu() const;

        template<size_t Xdim, size_t Ydim>
        inline static_matrix<ElemType, Xdim, Ydim>
        copy_to_cpu() const;

        inline ~gpu_matrix();
    };


}

#endif
