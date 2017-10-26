
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/host/matrix.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/system/exception.hpp>

#include <algorithm>

namespace mgcpp
{
    template<typename T,
             storage_order SO>
    host_matrix<T, SO>::
    host_matrix() noexcept
    : _data(nullptr),
        _released(true),
        _m_dim(0),
        _n_dim(0) {}

    template<typename T,
             storage_order SO>
    host_matrix<T, SO>::
    host_matrix(size_t i, size_t j)
        : _data(nullptr),
          _released(true),
          _m_dim(i),
          _n_dim(j)
    {
        size_t total_size = i * j;
        T* ptr =
            (T*)malloc(sizeof(T) * total_size);

        if(!ptr)
        {
            MGCPP_THROW_BAD_ALLOC;
        }
        _released = true;
        _data = ptr;
    }

    template<typename T,
             storage_order SO>
    host_matrix<T, SO>::
    host_matrix(size_t i, size_t j, T* data) noexcept
        : _data(data),
          _released(false),
          _m_dim(i),
          _n_dim(j) {}

    template<typename T,
             storage_order SO>
    host_matrix<T, SO>::
    host_matrix(size_t i, size_t j, T init)
        : _data(nullptr),
          _released(true),
          _m_dim(i),
          _n_dim(j)
    {
        size_t total_size = i * j;
        T* ptr = (T*)malloc(sizeof(T) * total_size);

        if(!ptr)
        {
            MGCPP_THROW_BAD_ALLOC;
        }

        std::fill(ptr, ptr + total_size, init);
        
        _data = ptr;
        _released = false;
    }

    template<typename T,
             storage_order SO>
    host_matrix<T, SO>::
    host_matrix(host_matrix<T, SO> const& other)
        : _data(nullptr),
          _released(true),
          _m_dim(other._m_dim),
          _n_dim(other._n_dim)
    {
        size_t total_size = other._m_dim * other._n_dim;

        T* ptr = (T*)malloc(sizeof(T) * total_size);
        if(!ptr)
        {
            MGCPP_THROW_BAD_ALLOC;
        }

        std::copy(ptr,ptr + total_size, other._data);
        _data = ptr;
        _released = false;
    }

    template<typename T,
             storage_order SO>
    host_matrix<T, SO>::
    host_matrix(host_matrix<T, SO>&& other) noexcept
        : _data(other._data),
          _released(false),
          _m_dim(other._m_dim),
          _n_dim(other._n_dim)
    {
        other._m_dim = 0;
        other._n_dim = 0;
        other._released = true;
    } 

    template<typename T,
             storage_order SO>
    template<size_t DeviceId>
    host_matrix<T, SO>::
    host_matrix(device_matrix<T, DeviceId, SO> const& gpu_mat)
        : _data(nullptr),
          _released(true),
          _m_dim(0),
          _n_dim(0)
    {
        auto shape = gpu_mat.shape();
        size_t total_size = shape.first * shape.second;

        T* host_memory = (T*)malloc(total_size * sizeof(T));
        if(!host_memory)
        {
            MGCPP_THROW_BAD_ALLOC;
        }

        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }        

        auto device_memory = gpu_mat.data();

        auto cpy_result =
            cuda_memcpy(host_memory, device_memory,
                        total_size,
                        cuda_memcpy_kind::device_to_host);
        if(!cpy_result)
        {
            free(host_memory);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }
        _data = host_memory;
        _released = false;
    }

    template<typename T,
             storage_order SO>
    host_matrix<T, SO>&
    host_matrix<T, SO>::
    operator=(host_matrix<T, SO> const& other)
    {
        if(!_released) 
        {
            free(_data);
            _released = true;
        }

        size_t total_size = other._m_dim * other._n_dim;

        T* ptr = (T*)malloc(sizeof(T) * total_size);
        if(!ptr)
        {
            MGCPP_THROW_BAD_ALLOC;
        }
        _released = false;

        std::copy(ptr,ptr + total_size, other._data);

        _data = ptr;
        _m_dim = other._m_dim;
        _n_dim = other._n_dim;

        return *this;
    }

    template<typename T,
             storage_order SO>
    host_matrix<T, SO>&
    host_matrix<T, SO>::
    operator=(host_matrix<T, SO>&& other) noexcept
    {
        if(!_released) 
        {
            free(_data);
            _released = true;
        }

        _data = other._data;
        _m_dim = other._m_dim;
        _n_dim = other._n_dim;
        _released = false;
        other._released = true;

        return *this;
    }

    template<typename T,
             storage_order SO>
    template<size_t DeviceId>
    host_matrix<T, SO>&
    host_matrix<T, SO>::
    operator=(device_matrix<T, DeviceId, SO> const& other)
    {
        if(!_released) 
        {
            free(_data);
            _released = true;
        }

        auto shape = other.shape();
        size_t total_size = shape.first * shape.second;

        T* host_memory = (T*)malloc(total_size * sizeof(T));
        if(!host_memory)
        {
            MGCPP_THROW_BAD_ALLOC;
        }

        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }        

        auto device_memory = other.data(); 

        auto cpy_result =
            cuda_memcpy(host_memory, device_memory,
                        total_size,
                        cuda_memcpy_kind::device_to_host);
        if(!cpy_result)
        {
            free(host_memory);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }
        _data = host_memory;
        _released = false;

        return *this;
    }

    template<typename T,
             storage_order SO>
    template<size_t DeviceId>
    device_matrix<T, DeviceId, SO>
    host_matrix<T, SO>::
    copy_to_device() const
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { 
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        size_t total_size = _m_dim * _n_dim;

        auto matrix =
            device_matrix<T, DeviceId, SO>(_m_dim, _n_dim);
        
        auto cpy_result =
            cuda_memcpy(matrix.data_mutable(),
                        _data,
                        total_size,
                        cuda_memcpy_kind::host_to_device);
        if(!cpy_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }
        return matrix;
    }


    template<typename T,
             storage_order SO>
    inline T
    host_matrix<T, SO>::
    at(size_t i, size_t j) const
    {
        if(i >= _m_dim || j >= _n_dim)
            MGCPP_THROW_OUT_OF_RANGE("index out of range");

        return _data[i * _n_dim + j];
    }

    template<typename T,
             storage_order SO>
    inline T&
    host_matrix<T, SO>::
    at(size_t i, size_t j) 
    {
        if(i >= _m_dim || j >= _n_dim)
            MGCPP_THROW_OUT_OF_RANGE("index out of range");

        return _data[i * _n_dim + j];
    }

    template<typename T,
             storage_order SO>
    inline T
    host_matrix<T, SO>::
    operator()(size_t i, size_t j) const noexcept
    {
        return _data[i * _n_dim + j];
    }

    template<typename T,
             storage_order SO>
    inline T&
    host_matrix<T, SO>::
    operator()(size_t i, size_t j) noexcept
    {
        return _data[i * _n_dim + j];
    }

    template<typename T,
             storage_order SO>
    inline std::pair<size_t, size_t> 
    host_matrix<T, SO>::
    shape() const noexcept
    {
        return {_m_dim, _n_dim};
    }

    template<typename T,
             storage_order SO>
    T const*
    host_matrix<T, SO>::
    data() const noexcept
    {
        return _data;
    }

    template<typename T,
             storage_order SO>
    T*
    host_matrix<T, SO>::
    data_mutable() noexcept
    {
        return _data;
    }

    template<typename T,
             storage_order SO> 
    T*
    host_matrix<T, SO>::
    release_data() noexcept
    {
        _released = true;
        return _data;
    }

    template<typename T,
             storage_order SO>
    host_matrix<T, SO>::
    ~host_matrix() noexcept
    {
        if(!_released)
            (void)free(_data);
    }
}
