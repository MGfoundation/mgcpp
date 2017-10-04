
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
             storage_order SO>
    gpu::matrix<ElemType, DeviceId, SO>::
    matrix() noexcept
    : _data(nullptr),
        _context(&global_context::get_thread_context()),
        _m_dim(0),
        _n_dim(0),
        _released(true) {}

    template<typename ElemType,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<ElemType, DeviceId, SO>::
    matrix(size_t i, size_t j)
        :_data(nullptr),
         _context(&global_context::get_thread_context()),
         _m_dim(i),
         _n_dim(j),
         _released(true)
    {
        auto result =
            cuda_malloc<ElemType>(_m_dim * _n_dim);
        if(!result)
            MGCPP_THROW_SYSTEM_ERROR(result.error());

        _released = false;
        _data = result.value();
    }


    template<typename ElemType,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<ElemType, DeviceId, SO>::
    matrix(size_t i, size_t j, ElemType init)
        :_data(nullptr),
         _context(&global_context::get_thread_context()),
         _m_dim(i),
         _n_dim(j),
         _released(true)
    {
        size_t total_size = _m_dim * _n_dim;
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
             storage_order SO>
    gpu::matrix<ElemType, DeviceId, SO>::
    matrix(cpu::matrix<ElemType, SO> const& cpu_mat)
        :_data(nullptr),
         _context(&global_context::get_thread_context()),
         _m_dim(0),
         _n_dim(0),
         _released(true)
    {
        // if(SO == row_major)
        // {
        //     _n_dim = cpu_mat.columns();
        //     _m_dim = cpu_mat.rows();
        // }
        // else
        // {
        //     _n_dim = cpu_mat.rows();
        //     _m_dim = cpu_mat.columns();
        // }

        auto shape = cpu_mat.shape();
        _m_dim = shape.first;
        _n_dim = shape.second;

        size_t total_size = _m_dim * _n_dim;

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
             storage_order SO>
    gpu::matrix<ElemType, DeviceId, SO>&
    gpu::matrix<ElemType, DeviceId, SO>::
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
            cuda_malloc<ElemType>(_n_dim * _m_dim);
        if(!alloc_result)
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        _released = false;
        _m_dim = i;
        _n_dim = j;

        return *this;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<ElemType, DeviceId, SO>&
    gpu::matrix<ElemType, DeviceId, SO>::
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
        _m_dim = i;
        _n_dim = j;

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
             storage_order SO>
    gpu::matrix<ElemType, DeviceId, SO>&
    gpu::matrix<ElemType, DeviceId, SO>::
    zeros()
    {
        if(_released)
            MGCPP_THROW_RUNTIME_ERROR("gpu memory wasn't allocated");

        auto set_result = cuda_memset(_data,
                                      static_cast<ElemType>(0),
                                      _m_dim * _n_dim);
        if(!set_result)
            MGCPP_THROW_SYSTEM_ERROR(set_result.error());

        return *this;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order SO>
    ElemType
    gpu::matrix<ElemType, DeviceId, SO>::
    check_value(size_t i, size_t j) const 
    {
        if(i > _m_dim || j > _n_dim)
            MGCPP_THROW_OUT_OF_RANGE("index out of range");

        ElemType* from = (_data + (i * _n_dim + j));
        ElemType to;
        auto result = cuda_memcpy(
            &to, from, 1, cuda_memcpy_kind::device_to_host);

        if(!result)
            MGCPP_THROW_SYSTEM_ERROR(result.error());

        return to;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<ElemType, DeviceId, SO>&
    gpu::matrix<ElemType, DeviceId, SO>::
    copy_from_host(cpu::matrix<ElemType, SO> const& cpu_mat)
    {
        if(this->shape() != cpu_mat.shape())
        {
            MGCPP_THROW_RUNTIME_ERROR("dimensions not matching");
        }

        size_t total_size = _m_dim * _n_dim;

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

        return *this;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order SO>
    cpu::matrix<ElemType, SO>
    gpu::matrix<ElemType, DeviceId, SO>::
    copy_to_host() const
    {
        size_t total_size = _m_dim * _n_dim;

        ElemType* host_memory =
            (ElemType*)malloc(total_size * sizeof(ElemType));
        if(!host_memory)
            MGCPP_THROW_BAD_ALLOC;
        
        auto memcpy_result =
            cuda_memcpy(host_memory,
                        _data,
                        total_size,
                        cuda_memcpy_kind::device_to_host);

        if(!memcpy_result)
        {
            free(host_memory);
            MGCPP_THROW_SYSTEM_ERROR(memcpy_result.error());
        }

        return cpu::matrix<ElemType, SO>(_m_dim,
                                         _n_dim,
                                         host_memory);
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order SO>
    inline ElemType const*
    gpu::matrix<ElemType, DeviceId, SO>::
    get_data() const
    {
        return _data;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order SO>
    inline ElemType*
    gpu::matrix<ElemType, DeviceId, SO>::
    get_data_mutable()
    {
        return _data;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order SO>
    inline ElemType*
    gpu::matrix<ElemType, DeviceId, SO>::
    release_data()
    {
        _released = true;
        return _data;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order SO>
    inline thread_context*
    gpu::matrix<ElemType, DeviceId, SO>::
    get_thread_context() const noexcept
    {
        return _context;
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order SO>
    std::pair<size_t, size_t>
    gpu::matrix<ElemType, DeviceId, SO>::
    shape() const noexcept
    {
        return {_m_dim, _n_dim};
    }

    template<typename ElemType,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<ElemType, DeviceId, SO>::
    ~matrix() noexcept
    {
        if(!_released)
            (void)cuda_free(_data);
        global_context::reference_cnt_decr();
    }
}
