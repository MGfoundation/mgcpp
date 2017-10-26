
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/global_context.hpp>
#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/cpu/vector.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/device/vector.hpp>
#include <mgcpp/system/exception.hpp>

#include <algorithm>

namespace mgcpp
{
    template<typename T,
             size_t DeviceId,
             allignment Allign>
    device_vector<T, DeviceId, Allign>::
    device_vector() noexcept
    : _data(nullptr),
        _context(&global_context::get_thread_context()),
        _size(0),
        _released(true) {}

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    device_vector<T, DeviceId, Allign>::
    device_vector(size_t size)
        : _data(nullptr),
          _context(&global_context::get_thread_context()),
          _size(size),
          _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto result = cuda_malloc<T>(_size);
        if(!result)
        {
            MGCPP_THROW_SYSTEM_ERROR(result.error());
        }

        _released = false;
        _data = result.value();
    }
        

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    device_vector<T, DeviceId, Allign>::
    device_vector(size_t size, T init)
        : _data(nullptr),
          _context(&global_context::get_thread_context()),
          _size(size),
          _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto result = cuda_malloc<T>(_size);
        if(!result)
        {
            MGCPP_THROW_SYSTEM_ERROR(result.error());
        }

        _released = false;
        _data = result.value();

        T* buffer = (T*)malloc(sizeof(T) * _size);
        if(!buffer)
        {
            MGCPP_THROW_BAD_ALLOC;
        }

        std::fill(buffer, buffer + _size, init);

        auto cpy_result =
            cuda_memcpy(_data, buffer, _size,
                        cuda_memcpy_kind::host_to_device);

        free(buffer);
        if(!cpy_result)
        {
            (void)cuda_free(_data); 
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    device_vector<T, DeviceId, Allign>::
    device_vector(device_vector<T, DeviceId, Allign> const& other)
        : _data(nullptr),
          _context(&global_context::get_thread_context()),
          _size(0),
          _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto alloc_result = cuda_malloc<T>(other._size);
        if(!alloc_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        }
        _released = false;
        _size = other._size;

        auto cpy_result =
            cuda_memcpy(alloc_result.value(),
                        other._data,
                        _size,
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
             allignment Allign>
    device_vector<T, DeviceId, Allign>::
    device_vector(device_vector<T, DeviceId, Allign>&& other) noexcept
        :_data(other._data),
         _context(&global_context::get_thread_context()),
         _size(other._size),
         _released(false)
    {
        other._released = true;
        other._size = 0;
        other._data = nullptr;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    device_vector<T, DeviceId, Allign>&
    device_vector<T, DeviceId, Allign>::
    operator=(device_vector<T, DeviceId, Allign> const& other)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        if(!_released)
        {
            auto free_result = cuda_free(_data);
            if(!free_result)
            {
                MGCPP_THROW_SYSTEM_ERROR(free_result.error());
            }
            _released = true;
        }

        auto alloc_result = cuda_malloc<T>(other._size);
        if(!alloc_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        }
        _released = false;
        _size = other._size;

        auto cpy_result =
            cuda_memcpy(alloc_result.value(),
                        other._data,
                        _size,
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
             allignment Allign>
    device_vector<T, DeviceId, Allign>&
    device_vector<T, DeviceId, Allign>::
    operator=(device_vector<T, DeviceId, Allign>&& other) noexcept
    {
        if(!_released)
        {
            (void)cuda_set_device(DeviceId);
            (void)cuda_free(_data);
        }
        _data = other._data;
        _released = false;
        _size = other._size;

        other._size = 0;
        other._data = nullptr;
        other._released = true;

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    device_vector<T, DeviceId, Allign>&
    device_vector<T, DeviceId, Allign>::
    zero()
    {
        if(_released)
        {
            MGCPP_THROW_RUNTIME_ERROR("gpu memory wasn't allocated");
        }

        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto set_result = cuda_memset(_data,
                                      static_cast<T>(0),
                                      _size);
        if(!set_result)
        { 
            MGCPP_THROW_SYSTEM_ERROR(set_result.error());
        }

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    cpu::vector<T, Allign> 
    device_vector<T, DeviceId, Allign>::
    copy_to_host() const
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        T* host_memory = (T*)malloc(_size * sizeof(T));
        if(!host_memory)
        {
            MGCPP_THROW_BAD_ALLOC;
        }
        
        auto cpy_result =
            cuda_memcpy(host_memory, _data, _size,
                        cuda_memcpy_kind::device_to_host);
        if(!cpy_result)
        {
            free(host_memory);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        return cpu::vector<T, Allign>(_size, host_memory);
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    void
    device_vector<T, DeviceId, Allign>::
    copy_from_host(cpu::vector<T, Allign> const& host) 
    {
        if(this->shape() != host.shape())
        {
            MGCPP_THROW_RUNTIME_ERROR("dimensions not matching");
        }
        if(_released)
        {
            MGCPP_THROW_RUNTIME_ERROR("memory not allocated");
        }

        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto cpy_result =
            cuda_memcpy(_data, host.get_data(), _size,
                        cuda_memcpy_kind::host_to_device);
        if(!cpy_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    T
    device_vector<T, DeviceId, Allign>::
    check_value(size_t i) const
    {
        if(i >= _size)
            MGCPP_THROW_OUT_OF_RANGE("index out of range");

        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        T* from = (_data + i);
        T to;
        auto result = cuda_memcpy(
            &to, from, 1, cuda_memcpy_kind::device_to_host);

        if(!result)
            MGCPP_THROW_SYSTEM_ERROR(result.error());

        return to;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    inline T*
    device_vector<T, DeviceId, Allign>::
    release_data() noexcept
    {
        _released = true;
        return _data;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    inline thread_context*
    device_vector<T, DeviceId, Allign>::
    context() const noexcept
    {
        return _context;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    T const*
    device_vector<T, DeviceId, Allign>::
    data() const noexcept
    {
        return _data;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    T*
    device_vector<T, DeviceId, Allign>::
    data_mutable() noexcept
    {
        return _data;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    size_t
    device_vector<T, DeviceId, Allign>::
    shape() const noexcept
    {
        return _size;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    size_t
    device_vector<T, DeviceId, Allign>::
    size() const noexcept
    {
        return _size;
    }
    
    template<typename T,
             size_t DeviceId,
             allignment Allign>
    device_vector<T, DeviceId, Allign>::
    ~device_vector() noexcept
    {
        (void)cuda_set_device(DeviceId);

        if(!_released)
            (void)cuda_free(_data);
        global_context::reference_cnt_decr();
    }
}
