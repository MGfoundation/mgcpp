
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/gpu/matrix.hpp>
#include <mgcpp/cpu/matrix.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cublas/cublas_helpers.hpp>

#include <cstdlib>
#include <type_traits>

namespace mgcpp
{
    template<typename T,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<T, DeviceId, SO>::
    matrix() noexcept
    : _data(nullptr),
        _context(&global_context::get_thread_context()),
        _m_dim(0),
        _n_dim(0),
        _released(true) {}

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<T, DeviceId, SO>::
    matrix(size_t i, size_t j)
        :_data(nullptr),
         _context(&global_context::get_thread_context()),
         _m_dim(i),
         _n_dim(j),
         _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());

        auto result = cuda_malloc<T>(_m_dim * _n_dim);
        if(!result)
        {
            MGCPP_THROW_SYSTEM_ERROR(result.error());
        }

        _released = false;
        _data = result.value();
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<T, DeviceId, SO>::
    matrix(size_t i, size_t j, T init)
        :_data(nullptr),
         _context(&global_context::get_thread_context()),
         _m_dim(i),
         _n_dim(j),
         _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());

        size_t total_size = _m_dim * _n_dim;
        auto alloc_result = cuda_malloc<T>(total_size);
        if(!alloc_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        }

        _released = false;
        _data = alloc_result.value();

        T* buffer = (T*)malloc(sizeof(T) * total_size);
        if(!buffer)
        {
            MGCPP_THROW_BAD_ALLOC;
        }

        std::fill(buffer, buffer + total_size, init);
        auto cpy_result = cublas_set_matrix(_n_dim, _m_dim,
                                            buffer, _data);

        free(buffer);
        if(!cpy_result)
        {
            (void)cuda_free(_data); 
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<T, DeviceId, SO>::
    matrix(gpu::matrix<T, DeviceId, SO> const& other)
        :_data(nullptr),
         _context(&global_context::get_thread_context()),
         _m_dim(0),
         _n_dim(0),
         _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());

        auto alloc_result =
            cuda_malloc<T>(other._m_dim * other._n_dim);
        if(!alloc_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        }
        _released = false;
        _m_dim = other._m_dim;
        _n_dim = other._n_dim;

        auto cpy_result =
            cuda_memcpy(alloc_result.value(),
                        other._data,
                        _m_dim * _n_dim,
                        cuda_memcpy_kind::device_to_device);

        if(!cpy_result)
        {
            (void)cuda_free(alloc_result.value());
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        _data = alloc_result.value();
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<T, DeviceId, SO>::
    matrix(cpu::matrix<T, SO> const& cpu_mat)
        :_data(nullptr),
         _context(&global_context::get_thread_context()),
         _m_dim(0),
         _n_dim(0),
         _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());

        auto shape = cpu_mat.shape();
        _m_dim = shape.first;
        _n_dim = shape.second;

        size_t total_size = _m_dim * _n_dim;

        auto alloc_result = cuda_malloc<T>(total_size);
        if(!alloc_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        }
        _released = false;
        
        auto cpy_result = cublas_set_matrix(_n_dim, _m_dim,
                                            cpu_mat.get_data(),
                                            _data);
        if(!cpy_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        _data = alloc_result.value();
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<T, DeviceId, SO>::
    matrix(gpu::matrix<T, DeviceId, SO>&& other) noexcept
        :_data(other._data),
         _context(&global_context::get_thread_context()),
         _m_dim(other._m_dim),
         _n_dim(other._n_dim),
         _released(false)
    {
        other._released = true;
        other._data = nullptr;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<T, DeviceId, SO>&
    gpu::matrix<T, DeviceId, SO>::
    operator=(gpu::matrix<T, DeviceId, SO> const& other)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());

        if(!_released)
        {
            auto free_result = cuda_free(_data);
            if(!free_result)
                MGCPP_THROW_SYSTEM_ERROR(free_result.error());
            _released = true;
        }

        auto alloc_result =
            cuda_malloc<T>(other._m_dim * other._n_dim);
        if(!alloc_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        }
        _released = false;
        _m_dim = other._m_dim;
        _n_dim = other._n_dim;

        auto cpy_result =
            cuda_memcpy(alloc_result.value(),
                        other._data,
                        _m_dim * _n_dim,
                        cuda_memcpy_kind::device_to_device);

        if(!cpy_result)
        {
            (void)cuda_free(alloc_result.value());
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }
        _data = alloc_result.value();

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<T, DeviceId, SO>&
    gpu::matrix<T, DeviceId, SO>::
    operator=(gpu::matrix<T, DeviceId, SO>&& other) noexcept
    {
        if(!_released)
            (void)cuda_free(_data);
        _data = other._data;
        _m_dim = other._m_dim;
        _n_dim = other._n_dim;

        other._data = nullptr;
        other._released = true;

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<T, DeviceId, SO>&
    gpu::matrix<T, DeviceId, SO>::
    resize(size_t i, size_t j)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());

        if(!_released)
        {
            auto free_result = cuda_free(_data);
            _released = true;
            if(!free_result)
                MGCPP_THROW_SYSTEM_ERROR(free_result.error());
        }

        auto alloc_result =
            cuda_malloc<T>(_n_dim * _m_dim);

        if(!alloc_result)
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        _released = false;
        _m_dim = i;
        _n_dim = j;

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<T, DeviceId, SO>&
    gpu::matrix<T, DeviceId, SO>::
    resize(size_t i, size_t j, T init)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());

        auto free_result = cuda_free(_data);
        _released = true;
        if(!free_result)
            MGCPP_THROW_SYSTEM_ERROR(free_result.error());

        size_t total_size = i * j;

        auto alloc_result = cuda_malloc<T>(total_size);
        if(!alloc_result)
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        _released = false;
        _m_dim = i;
        _n_dim = j;

        T* buffer = (T*)malloc(sizeof(T) * total_size);
        if(!buffer)
            MGCPP_THROW_BAD_ALLOC;

        std::fill(buffer, buffer + total_size, init);
        auto cpy_result = cublas_set_matrix(_n_dim, _m_dim,
                                            buffer, _data);
        free(buffer);
        if(!cpy_result)
        {
            // (void)free_pinned(buffer_result.value());
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<T, DeviceId, SO>&
    gpu::matrix<T, DeviceId, SO>::
    zeros()
    {
        if(_released)
            MGCPP_THROW_RUNTIME_ERROR("gpu memory wasn't allocated");

        auto set_result = cuda_memset(_data,
                                      static_cast<T>(0),
                                      _m_dim * _n_dim);
        if(!set_result)
            MGCPP_THROW_SYSTEM_ERROR(set_result.error());

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    T
    gpu::matrix<T, DeviceId, SO>::
    check_value(size_t i, size_t j) const 
    {
        if(i >= _m_dim || j >= _n_dim)
            MGCPP_THROW_OUT_OF_RANGE("index out of range");

        T* from = (_data + (i * _n_dim + j));
        T to;
        auto result = cuda_memcpy(
            &to, from, 1, cuda_memcpy_kind::device_to_host);

        if(!result)
            MGCPP_THROW_SYSTEM_ERROR(result.error());

        return to;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<T, DeviceId, SO>&
    gpu::matrix<T, DeviceId, SO>::
    copy_from_host(cpu::matrix<T, SO> const& cpu_mat)
    {
        if(this->shape() != cpu_mat.shape())
        {
            MGCPP_THROW_RUNTIME_ERROR("dimensions not matching");
        }
        if(_released)
        {
            MGCPP_THROW_RUNTIME_ERROR("memory not allocated");
        }

        auto cpy_result = cublas_set_matrix(_n_dim, _m_dim,
                                            cpu_mat.get_data(),
                                            _data);

        if(!cpy_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    cpu::matrix<T, SO>
    gpu::matrix<T, DeviceId, SO>::
    copy_to_host() const
    {
        size_t total_size = _m_dim * _n_dim;

        T* host_memory = (T*)malloc(total_size * sizeof(T));
        if(!host_memory)
        {
            MGCPP_THROW_BAD_ALLOC;
        }
        
        auto cpy_result = cublas_get_matrix(_n_dim, _m_dim,
                                            _data, host_memory);

        if(!cpy_result)
        {
            free(host_memory);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        return cpu::matrix<T, SO>(_m_dim,
                                  _n_dim,
                                  host_memory);
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    inline T const*
    gpu::matrix<T, DeviceId, SO>::
    get_data() const noexcept
    {
        return _data;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    inline T*
    gpu::matrix<T, DeviceId, SO>::
    get_data_mutable() noexcept
    {
        return _data;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    inline T*
    gpu::matrix<T, DeviceId, SO>::
    release_data()
    {
        _released = true;
        return _data;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    inline thread_context*
    gpu::matrix<T, DeviceId, SO>::
    get_thread_context() const noexcept
    {
        return _context;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    std::pair<size_t, size_t>
    gpu::matrix<T, DeviceId, SO>::
    shape() const noexcept
    {
        return {_m_dim, _n_dim};
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    gpu::matrix<T, DeviceId, SO>::
    ~matrix() noexcept
    {
        (void)cuda_set_device(DeviceId);

        if(!_released)
            (void)cuda_free(_data);
        global_context::reference_cnt_decr();
    }
}
