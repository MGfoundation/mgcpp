#ifndef _DECLARATIONS_HPP_
#define _DECLARATIONS_HPP_

namespace mg
{

    template<typename ElemType, size_t DeviceId>
    class gpu_matrix;

    template<typename ElemType, size_t Xdim, size_t Ydim>
    class static_matrix;

    template<typename ElemType>
    class dynamic_matrix;

    template<typename ElemType, size_t DeviceId>
    using gmat = gpu_matrix<ElemType, DeviceId>;

    template<typename ElemType, size_t Xdim, size_t Ydim>
    using smat = static_matrix<ElemType, Xdim, Ydim>;

    template<typename ElemType>
    using dmat = dynamic_matrix<ElemType>;
}

#endif
