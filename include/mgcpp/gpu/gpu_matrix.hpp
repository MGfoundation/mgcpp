#ifndef _GPU_MATRIX_HPP_
#define _GPU_MATRIX_HPP_

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
            size_t _id;

        public:
            inline matrix();

            inline matrix(size_t x_dim, size_t y_dim);

            inline matrix(size_t x_dim, size_t y_dim, ElemType init);

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

            inline ~matrix();
        };
    }
}

#endif
