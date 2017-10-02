
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/gpu/matrix.hpp>
#include <mgcpp/cpu/matrix.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/cublas/cublas_helpers.hpp>

#include <cstdlib>
#include <type_traits>
#include <cstring>

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

        _released = false;
        _data = result.value();
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

        _released = false;
        _data = result.value();
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
        size_t total_size = i * j;
        auto alloc_result = cuda_malloc<ElemType>(total_size);
        if(!alloc_result)
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());

        _released = false;
        _data = alloc_result.value();

        ElemType* buffer =
            (ElemType*)malloc(sizeof(ElemType) * total_size);
        if(!buffer)
            MGCPP_THROW_BAD_ALLOC;

        memset(buffer, init, sizeof(ElemType) * total_size);
        
        auto memcpy_result =
            cuda_memcpy(_data,
                        buffer,
                        total_size,
                        cuda_memcpy_kind::host_to_device);
        free(buffer);
        if(!memcpy_result)
        {
            // (void)free_pinned(buffer_result.value());
            MGCPP_THROW_SYSTEM_ERROR(memcpy_result.error());
        }
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
    matrix(cpu::matrix<ElemType, StoreOrder> const& cpu_mat)
        :_data(nullptr),
         _context(nullptr),
         _row_dim(cpu_mat.rows()),
         _col_dim(cpu_mat.columns()),
         _released(true)
    {
        size_t total_size = _row_dim * _col_dim;

        auto alloc_result = cuda_malloc<ElemType>(total_size);
        if(!alloc_result)
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        _released = false;

        _data = alloc_result.value();
        
        auto memcpy_result =
            cuda_memcpy(_data,
                        cpu_mat.get_data(),
                        total_size,
                        cuda_memcpy_kind::host_to_device);

        if(!memcpy_result)
        {
            // (void)free_pinned(buffer_result.value());
            MGCPP_THROW_SYSTEM_ERROR(memcpy_result.error());
        }
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    gpu::matrix<ElemType, DeviceId, StoreOrder>&
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    resize(size_t i, size_t j)
    {
        if(!_released)
        {
            auto free_result = cuda_free(_data);
            _released = true;
            if(!free_result)
                MGCPP_THROW_SYSTEM_ERROR(free_result.error());
        }

        auto alloc_result =
            cuda_malloc<ElemType>(_row_dim * _col_dim);
        if(!alloc_result)
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        _released = false;
        _col_dim = i;
        _row_dim = j;

        return *this;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    gpu::matrix<ElemType, DeviceId, StoreOrder>&
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    resize(size_t i, size_t j, ElemType init)
    {
        auto free_result = cuda_free(_data);
        _released = true;
        if(!free_result)
            MGCPP_THROW_SYSTEM_ERROR(free_result.error());

        size_t total_size = i * j;

        auto alloc_result = cuda_malloc<ElemType>(total_size);
        if(!alloc_result)
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        _released = false;
        _col_dim = i;
        _row_dim = j;

        // auto buffer_result = malloc_pinned<ElemType>(i * j);

        ElemType* buffer =
            (ElemType*)malloc(sizeof(ElemType) * total_size);
        if(!buffer)
            MGCPP_THROW_BAD_ALLOC;

        memset(buffer, init, sizeof(ElemType) * total_size);
        
        auto memcpy_result =
            cuda_memcpy(_data,
                        buffer,
                        total_size,
                        cuda_memcpy_kind::host_to_device);
        free(buffer);
        if(!memcpy_result)
        {
            // (void)free_pinned(buffer_result.value());
            MGCPP_THROW_SYSTEM_ERROR(memcpy_result.error());
        }

        return *this;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    gpu::matrix<ElemType, DeviceId, StoreOrder>&
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    zeros()
    {
        if(_released)
            MGCPP_THROW_RUNTIME_ERROR("gpu memory wasn't allocated");

        auto set_result = cuda_memset(_data,
                                      static_cast<ElemType>(0),
                                      _row_dim * _col_dim);
        if(!set_result)
            MGCPP_THROW_SYSTEM_ERROR(set_result.error());

        return *this;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    ElemType
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    check_value(size_t i, size_t j) const 
    {
        if(i > _col_dim || j > _row_dim)
            MGCPP_THROW_OUT_OF_RANGE("index out of range");

        ElemType* from = (_data + (i * _row_dim + j));
        ElemType to;
        auto result = cuda_memcpy(
            &to, from, 1, cuda_memcpy_kind::device_to_host);

        if(!result)
            MGCPP_THROW_SYSTEM_ERROR(result.error());

        return to;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    size_t
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    columns() const noexcept
    {
        return StoreOrder == storage_order::column_major
            ? _col_dim : _row_dim;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    size_t
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    rows() const noexcept
    {
        return StoreOrder == storage_order::column_major
            ? _row_dim : _col_dim;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    gpu::matrix<ElemType, DeviceId, StoreOrder>::
    ~matrix() noexcept
    {
        if(!_released)
            (void)cuda_free(_data);
    }
}
