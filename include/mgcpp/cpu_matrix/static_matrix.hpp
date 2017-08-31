#ifndef _STATIC_MATRIX_HPP_
#define _STATIC_MATRIX_HPP_

#include <mgcpp/utility/declarations.hpp>

namespace mg
{
    template<typename ElemType,
             size_t Xdim, size_t Ydim>
    class static_matrix
    {
    public:
        inline static_matrix();

        inline static_matrix(ElemType init);

        template<size_t DeviceId>
        inline static_matrix(
            gmat<ElemType, DeviceId> const& gpu_mat);

        template<size_t DeviceId>
        inline gmat<ElemType, DeviceId>
        copy_to_gpu();
    };
}

#endif
