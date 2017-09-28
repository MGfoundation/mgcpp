
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/gpu/matrix.hpp>
#include <mgcpp/cpu/matrix.hpp>
#include <mgcpp/cuda/cuda_template_stdlib.hpp>
#include <mgcpp/cublas/cublas_helpers.hpp>

namespace mg
{

    template<typename ElemType, size_t DeviceId>
    gpu::matrix<ElemType, DeviceId>::
    matrix()
        : data(nullptr),
          _x_dim(0),
          _y_dim(0),
          _released(true)
    {}

    template<typename ElemType, size_t DeviceId>
    gpu::matrix<ElemType, DeviceId>::
    matrix(size_t x_dim, size_t y_dim)
        : data(cuda_malloc<ElemType>(x_dim * y_dim)),
          _x_dim(x_dim),
          _y_dim(y_dim),
          _released(false)
    {}

    template<typename ElemType, size_t DeviceId>
    gpu::matrix<ElemType, DeviceId>::
    matrix(size_t x_dim, size_t y_dim, ElemType init)
        : data(cuda_malloc<ElemType>(x_dim * y_dim)),
          _x_dim(x_dim),
          _y_dim(y_dim),
          _released(false)
    {

    }

    template<typename ElemType, size_t DeviceId>
    gpu::matrix<ElemType, DeviceId>::
    matrix(cpu::matrix<ElemType> const& cpu_mat)
    {
        auto rows = cpu_mat.rows();
        auto cols = cpu_mat.columns();
        _data = cuda_malloc<ElemType>(rows * cols);
        cuabls_set_matrix();
    }

    template<typename ElemType, size_t DeviceId>
    size_t 
    gpu::matrix<ElemType, DeviceId>::
    rows() const noexcept
    {
        return _row_dim;
    }

    template<typename ElemType, size_t DeviceId>
    size_t
    gpu::matrix<ElemType, DeviceId>::
    columns() const noexcept
    {
        return _col_dim;
    }
}
