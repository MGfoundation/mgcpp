
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/host/vector.hpp>
#include <mgcpp/device/vector.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    template<typename T,
             allignment Allign>
    host_vector<T, Allign>::
    host_vector() noexcept
    : _data(nullptr),
        _size(0),
        _released(true) {}
    
    template<typename T,
             allignment Allign>
    host_vector<T, Allign>::
    host_vector(size_t size) 
        : _data(nullptr),
          _size(size),
          _released(true)
    {
        T* ptr = (T*)malloc(sizeof(T) * _size);
        if(!ptr)
        {
            MGCPP_THROW_BAD_ALLOC;
        }
        _released = false;
        
        _data = ptr;
    }

    template<typename T,
             allignment Allign>
    host_vector<T, Allign>::
    host_vector(size_t size, T init)
        : _data(nullptr),
          _size(size),
          _released(true)
    {
        T* ptr = (T*)malloc(sizeof(T) * _size);
        if(!ptr)
        {
            MGCPP_THROW_BAD_ALLOC;
        }
        _released = false;

        std::fill(ptr, ptr + _size, init);
        
        _data = ptr;
    }

    template<typename T,
             allignment Allign>
    host_vector<T, Allign>::
    host_vector(size_t size, T* data) noexcept
        : _data(data),
          _size(size), 
          _released(false) {}

    template<typename T,
             allignment Allign>
    host_vector<T, Allign>::
    host_vector(host_vector<T, Allign> const& other)
        : _data(nullptr),
          _size(other.size()), 
          _released(true) 
    {
        T* ptr = (T*)malloc(sizeof(T) * _size);
        if(!ptr)
        {
            MGCPP_THROW_BAD_ALLOC;
        }
        _released = false;

        std::copy(ptr, ptr + _size, other._data);
        
        _data = ptr;
    }

    template<typename T,
             allignment Allign>
    host_vector<T, Allign>::
    host_vector(std::initializer_list<T> const& array) noexcept
    {
        
    }

    template<typename T,
             allignment Allign>
    host_vector<T, Allign>::
    host_vector(host_vector<T, Allign>&& other) noexcept
        : _data(other._data),
          _size(other._size), 
          _released(false) 
    {
        other._size = 0;
        other._data = nullptr;
        other._released = true;
    }

    template<typename T,
             allignment Allign>
    template<size_t DeviceId>
    host_vector<T, Allign>::
    host_vector(device_vector<T, DeviceId, Allign> const& other) 
        : _data(nullptr),
          _size(other._size),
          _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto* device_memory = other.data();

        T* host_memory = (T*)malloc(_size * sizeof(T));
        if(!host_memory)
        {
            MGCPP_THROW_BAD_ALLOC;
        }
        _released = false;
        
        auto cpy_result =
            cuda_memcpy(host_memory, device_memory, _size,
                        cuda_memcpy_kind::device_to_host);
        if(!cpy_result)
        {
            free(host_memory);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        _data = host_memory;
    }

    template<typename T,
             allignment Allign>
    host_vector<T, Allign>&
    host_vector<T, Allign>::
    operator=(host_vector<T, Allign> const& other)
    {
        if(!_released)
        {
            free(_data);
            _released = true;
        }

        T* ptr = (T*)malloc(sizeof(T) * _size);
        if(!ptr)
        {
            MGCPP_THROW_BAD_ALLOC;
        }
        _released = false;

        std::copy(ptr, ptr + _size, other._data);
        
        _data = ptr;
    }

    template<typename T,
             allignment Allign>
    host_vector<T, Allign>&
    host_vector<T, Allign>::
    operator=(host_vector<T, Allign>&& other) noexcept
    {
        if(!_released)
        {
            free(_data);
            _released = true;
        }

        _data = other._data;
        _size = other._size ;
        _released = false;

        other._size = 0;
        other._data = nullptr;
        other._released = true;

    }

    template<typename T,
             allignment Allign>
    template<size_t DeviceId>
    host_vector<T, Allign>&
    host_vector<T, Allign>::
    operator=(device_vector<T, DeviceId, Allign> const& other)
    {
        if(!_released)
        {
            free(_data);
            _released = true;
        }

        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto* device_memory = other.data();

        T* host_memory = (T*)malloc(_size * sizeof(T));
        if(!host_memory)
        {
            MGCPP_THROW_BAD_ALLOC;
        }
        _released = false;
        
        auto cpy_result =
            cuda_memcpy(host_memory, device_memory, _size,
                        cuda_memcpy_kind::device_to_host);
        if(!cpy_result)
        {
            free(host_memory);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        _data = host_memory;
    }

    template<typename T,
             allignment Allign>
    template<size_t DeviceId>
    device_vector<T, DeviceId, Allign>
    host_vector<T, Allign>::
    copy_to_gpu() const
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto device_vec =
            device_vector<T, DeviceId, Allign>(_size);

        auto cpy_result =
            cuda_memcpy(_data,
                        device_vec.data_mutable(),
                        _size,
                        cuda_memcpy_kind::host_to_device);
        if(!cpy_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        return device_vec;
    }

    template<typename T,
             allignment Allign>
    T
    host_vector<T, Allign>::
    at(size_t i) const
    {
        if(i >= _size)
        {
            MGCPP_THROW_OUT_OF_RANGE("index out of range");
        }

        return _data[i];
    }

    template<typename T,
             allignment Allign>
    T&
    host_vector<T, Allign>::
    at(size_t i)
    {
        if(i >= _size)
        {
            MGCPP_THROW_OUT_OF_RANGE("index out of range");
        }

        return _data[i];
    }

    template<typename T,
             allignment Allign>
    T
    host_vector<T, Allign>::
    operator[](size_t i) const noexcept
    {
        return _data[i];
    }

    template<typename T,
             allignment Allign>
    T&
    host_vector<T, Allign>::
    operator[](size_t i) noexcept
    {
        return _data[i];
    }

    template<typename T,
             allignment Allign>
    T const*
    host_vector<T, Allign>::
    data() const
    {
        return _data; 
    }

    template<typename T,
             allignment Allign>
    T*
    host_vector<T, Allign>::
    data_mutable() noexcept
    {
        return _data;
    }

    template<typename T,
             allignment Allign>
    size_t
    host_vector<T, Allign>::
    shape() const noexcept
    {
        return _size; 
    }

    template<typename T,
             allignment Allign>
    size_t
    host_vector<T, Allign>::
    size() const noexcept
    {
        return _size; 
    }

    template<typename T,
             allignment Allign>
    T*
    host_vector<T, Allign>::
    released_data()
    {
        _released= true;
        return _data;
    }


    template<typename T,
             allignment Allign>
    host_vector<T, Allign>::
    ~host_vector() noexcept
    {
        if(!_released)
            (void)free(_data);
    }
}
