
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/gpu/vector.hpp>
#include <mgcpp/cpu/vector.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cublas/cublas_helpers.hpp>

#include <algorithm>

namespace mgcpp
{
    template<typename T,
             size_t DeviceId,
             allignment Allign>
    gpu::vector<T, DeviceId, Allign>::
    vector() noexcept
    : _data(nullptr),
        _context(&global_context::get_thread_context()),
        _size(0),
        _released(true) {}

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    gpu::vector<T, DeviceId, Allign>::
    vector(size_t size)
        : _data(nullptr),
          _context(&global_context::get_thread_context()),
          _size(size),
          _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());

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
    gpu::vector<T, DeviceId, Allign>::
    vector(size_t size, T init)
        : _data(nullptr),
          _context(&global_context::get_thread_context()),
          _size(size),
          _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());

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
        auto cpy_result = cublas_set_vector(_size, buffer, _data);

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
    gpu::vector<T, DeviceId, Allign>::
    vector(size_t size, T* data)
    {
        
    }

    
    template<typename T,
             size_t DeviceId,
             allignment Allign>
    gpu::vector<T, DeviceId, Allign>::
    vector(gpu::vector<T, DeviceId, Allign> const& other)
        : _data(nullptr),
          _context(&global_context::get_thread_context()),
          _size(0),
          _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());

        auto alloc_result = cuda_malloc<T>(_size);
        if(!alloc_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        }
        _released = false;
        _size = other._size();

        auto cpy_result =
            cuda_memcpy(alloc_result.value(),
                        other._data,
                        size,
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
    gpu::vector<T, DeviceId, Allign>::
    vector(gpu::vector<T, DeviceId, Allign>&& other) noexcept
        :_data(other._data),
         _context(&global_context::get_thread_context()),
         _size(other._size),
         _released(false)
    {
        other._released = true;
        other._data = nullptr;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    gpu::vector<T, DeviceId, Allign>&
    gpu::vector<T, DeviceId, Allign>::
    operator=(gpu::vector<T, DeviceId, Allign> const& other)
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
            cuda_free(alloc_result.value());
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        _data = alloc_result.value();
        return *this;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    gpu::vector<T, DeviceId, Allign>&
    gpu::vector<T, DeviceId, Allign>::
    operator=(gpu::vector<T, DeviceId, Allign>&& other) noexcept
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
             allignment Allign>
    cpu::vector<T, Allign> 
    gpu::vector<T, DeviceId, Allign>::
    copy_to_host() const
    {
        T* host_memory = (T*)malloc(_size * sizeof(T));
        if(!host_memory)
        {
            MGCPP_THROW_BAD_ALLOC;
        }
        
        auto cpy_result = cublas_get_vector(_size, _data, host_memory);
        if(!cpy_result)
        {
            free(host_memory);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        return cpu::matrix<T, SO>(_size, host_memory);
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    void
    gpu::vector<T, DeviceId, Allign>::
    copy_from_host(cpu::vector<T, Allign> const& host) const
    {
        if(this->shape() != cpu_mat.shape())
        {
            MGCPP_THROW_RUNTIME_ERROR("dimensions not matching");
        }
        if(_released)
        {
            MGCPP_THROW_RUNTIME_ERROR("memory not allocated");
        }

        auto cpy_result = cublas_set_vector(_size,
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
             allignment Allign>
    T
    gpu::vector<T, DeviceId, Allign>::
    check_value(size_t i) const
    {
        if(i >= _size)
            MGCPP_THROW_OUT_OF_RANGE("index out of range");

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
    gpu::vector<T, DeviceId, Allign>::
    release_data() noexcept
    {
        _released = true;
        return _data;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    inline thread_context*
    gpu::vector<T, DeviceId, Allign>::
    get_thread_context() const noexcept
    {
        return _context;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    T const*
    gpu::vector<T, DeviceId, Allign>::
    get_data() const noexcept
    {
        return _data;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    T*
    gpu::vector<T, DeviceId, Allign>::
    get_data_mutable() noexcept
    {
        return _data;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    gpu::vector<T, DeviceId, Allign>::
    size_t
    shape() const noexcept
    {
        return _size;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign>
    gpu::vector<T, DeviceId, Allign>::
    size_t
    size() const noexcept
    {
        return _size;
    }
    
    template<typename T,
             size_t DeviceId,
             allignment Allign>
    gpu::vector<T, DeviceId, Allign>::
    ~vector() noexcept
    {
        (void)cuda_set_device(DeviceId);

        if(!_released)
            (void)cuda_free(_data);
        global_context::reference_cnt_decr();
    }
}
#endif
