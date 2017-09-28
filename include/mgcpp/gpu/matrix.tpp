
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/gpu/matrix.hpp>
#include <mgcpp/cpu/matrix.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/cuda/stdlib.hpp>
#include <mgcpp/cublas/cublas_helpers.hpp>

namespace mgcpp
{
    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    matrix() noexcept
    : _data(nullptr),
        _context(nullptr),
        _row_dim(0),
        _col_dim(0),
        _released(true) {}

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    matrix(thread_context& context) noexcept
        : _data(nullptr),
          _context(&context),
          _row_dim(0),
          _col_dim(0),
          _released(true) {}

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    matrix(size_t i, size_t j)
        :_data(nullptr),
         _context(nullptr),
         _row_dim(j),
         _col_dim(i),
         _released(true)
    {
        auto result =
            cuda_malloc<ElemType>(_row_dim * _col_dim);
        if(!result)
            MGCPP_THROW_SYSTEM_ERROR(result.error());
        else
        {
            _released = false;
            _data = result;
        }
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    matrix(thread_context& context,
           size_t i, size_t j)
        :_data(nullptr),
         _context(&context),
         _row_dim(j),
         _col_dim(i),
         _released(true)
    {
        auto result =
            cuda_malloc<ElemType>(_row_dim * _col_dim);
        if(!result)
            MGCPP_THROW_SYSTEM_ERROR(result.error());
        else
        {
            _released = false;
            _data = result.value();
        }
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    matrix(size_t i, size_t j, ElemType init)
        :_data(nullptr),
         _context(nullptr),
         _row_dim(j),
         _col_dim(i),
         _released(true)
    {
        auto alloc_result =
            cuda_malloc<ElemType>(_row_dim * _col_dim);
        if(!alloc_result)
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        else
        {
            _data = alloc_result.value();
        }

        auto set_result =
            cuda_memset(_data, init, _row_dim * _col_dim);
        if(!set_result)
            MGCPP_THROW_SYSTEM_ERROR(set_result.error());
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    matrix(thread_context& context,
           size_t i, size_t j, ElemType init)
        :_data(nullptr),
         _context(&context),
         _row_dim(j),
         _col_dim(i),
         _released(true)
    {
        auto alloc_result =
            cuda_malloc<ElemType>(_row_dim * _col_dim);
        if(!alloc_result)
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        else
        {
            _released = false;
            _data = alloc_result.value();
        }

        auto set_result =
            cuda_memset(_data, init, _row_dim * _col_dim);
        if(!set_result)
            MGCPP_THROW_SYSTEM_ERROR(set_result.error());
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    matrix(cpu::matrix<ElemType> const& cpu_mat)
    {
        auto rows = cpu_mat.rows();
        auto cols = cpu_mat.columns();
        auto alloc_result = cuda_malloc<ElemType>(rows * cols);
        if(!alloc_result)
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        
        // cubals_set_matrix(rows, cols, cpu_mat.get_data() ,_data);
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    ElemType
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    check_value(size_t i, size_t j) const noexcept
    {
        if(i > _col_dim || j > _row_dim)
            MGCPP_THROW_OUT_OF_RANGE("index out of range");
        return _data[i * _row_dim + j];
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    size_t
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    columns() const noexcept
    {
        return _col_dim;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    size_t
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    rows() const noexcept
    {
        return _row_dim;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    ~matrix()
    {
        if(!_released)
            (void)cuda_free(_data);
    }
}
