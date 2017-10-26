
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/device/matrix.hpp>
#include <mgcpp/host/matrix.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/cuda/device.hpp>

#include <cstdlib>
#include <type_traits>

namespace mgcpp
{
    template<typename T,
             size_t DeviceId,
             storage_order SO>
    device_matrix<T, DeviceId, SO>::
    device_matrix() noexcept
    : _data(nullptr),
        _context(&global_context::get_thread_context()),
        _shape(0, 0),
        _released(true) {}

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    device_matrix<T, DeviceId, SO>::
    device_matrix(size_t i, size_t j)
        :_data(nullptr),
         _context(&global_context::get_thread_context()),
         _shape(i, j),
         _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { 
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto result =
            cuda_malloc<T>(_shape.first * _shape.second);
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
    device_matrix<T, DeviceId, SO>::
    device_matrix(size_t i, size_t j, T init)
        :_data(nullptr),
         _context(&global_context::get_thread_context()),
         _shape(i, j),
         _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        size_t total_size = _shape.first * _shape.second;
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

        auto cpy_result =
            cuda_memcpy(_data, buffer,
                        _shape.first * _shape.second,
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
             storage_order SO>
    device_matrix<T, DeviceId, SO>::
    device_matrix(device_matrix<T, DeviceId, SO> const& other)
        :_data(nullptr),
         _context(&global_context::get_thread_context()),
         _shape(),
         _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { 
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto total_size = other._shape.first * other._shape.second;
        auto alloc_result = cuda_malloc<T>(total_size);
        if(!alloc_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        }
        _released = false;
        _shape = other._shape;

        auto cpy_result =
            cuda_memcpy(alloc_result.value(),
                        other._data,
                        _shape.first * _shape.second,
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
    device_matrix<T, DeviceId, SO>::
    device_matrix(host_matrix<T, SO> const& cpu_mat)
        :_data(nullptr),
         _context(&global_context::get_thread_context()),
         _shape(),
         _released(true)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { 
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto shape = cpu_mat.shape();
        _shape.first = shape.first;
        _shape.second = shape.second;

        size_t total_size = _shape.first * _shape.second;

        auto alloc_result = cuda_malloc<T>(total_size);
        if(!alloc_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        }
        _released = false;
        
        auto cpy_result =
            cuda_memcpy(alloc_result.value(),
                        cpu_mat.data(),
                        total_size,
                        cuda_memcpy_kind::host_to_device);
        if(!cpy_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        _data = alloc_result.value();
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    device_matrix<T, DeviceId, SO>::
    device_matrix(device_matrix<T, DeviceId, SO>&& other) noexcept
        :_data(other._data),
         _context(&global_context::get_thread_context()),
         _shape(std::move(other._shape)),
         _released(false)
    {
        other._released = true;
        other._data = nullptr;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    device_matrix<T, DeviceId, SO>&
    device_matrix<T, DeviceId, SO>::
    operator=(device_matrix<T, DeviceId, SO> const& other)
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

        auto total_size = other._shape.first * other._shape.second;
        auto alloc_result = cuda_malloc<T>(total_size);
        if(!alloc_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        }
        _released = false;
        _shape = other._shape;

        auto cpy_result =
            cuda_memcpy(alloc_result.value(),
                        other._data,
                        total_size,
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
    device_matrix<T, DeviceId, SO>&
    device_matrix<T, DeviceId, SO>::
    operator=(device_matrix<T, DeviceId, SO>&& other) noexcept
    {
        if(!_released)
        { 
            (void)cuda_set_device(DeviceId);
            (void)cuda_free(_data);
            _released = true;
        }
        _data = other._data;
        _released = false;
        other._data = nullptr;
        other._released = true;

        _shape = std::move(other._shape);

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    device_matrix<T, DeviceId, SO>&
    device_matrix<T, DeviceId, SO>::
    resize(size_t i, size_t j)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { 
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        if(!_released)
        {
            auto free_result = cuda_free(_data);
            _released = true;
            if(!free_result)
            { 
                MGCPP_THROW_SYSTEM_ERROR(free_result.error());
            }
        }

        auto alloc_result =
            cuda_malloc<T>(i * j);

        if(!alloc_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        }
        _released = false;
        _shape = std::make_pair(i, j);

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    device_matrix<T, DeviceId, SO>&
    device_matrix<T, DeviceId, SO>::
    resize(size_t i, size_t j, T init)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { 
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        auto free_result = cuda_free(_data);
        _released = true;
        if(!free_result)
        { 
            MGCPP_THROW_SYSTEM_ERROR(free_result.error());
        }

        size_t total_size = i * j;

        auto alloc_result = cuda_malloc<T>(total_size);
        if(!alloc_result)
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        _released = false;
        _shape = std::make_pair(i, j);

        T* buffer = (T*)malloc(sizeof(T) * total_size);
        if(!buffer)
        {
            MGCPP_THROW_BAD_ALLOC;
        }

        std::fill(buffer, buffer + total_size, init);

        auto cpy_result =
            cuda_memcpy(_data, buffer,
                        _shape.first * _shape.second,
                        cuda_memcpy_kind::host_to_device);
        free(buffer);
        if(!cpy_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        return *this;
    }

    template<typename T,
             size_t DeviceId,
                storage_order SO>
    host_matrix<T, SO>
    device_matrix<T, DeviceId, SO>::
    copy_to_host() const
    {
        size_t total_size = _shape.first * _shape.second;

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

        auto cpy_result =
            cuda_memcpy(host_memory, _data,
                        total_size,
                        cuda_memcpy_kind::device_to_host);

        if(!cpy_result)
        {
            free(host_memory);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        return host_matrix<T, SO>(_shape.first,
                                  _shape.second,
                                  host_memory);
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    device_matrix<T, DeviceId, SO>&
    device_matrix<T, DeviceId, SO>::
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

        auto set_result
            = cuda_memset(_data, static_cast<T>(0),
                          _shape.first * _shape.second);
        if(!set_result)
        { 
            MGCPP_THROW_SYSTEM_ERROR(set_result.error());
        }

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    T
    device_matrix<T, DeviceId, SO>::
    check_value(size_t i, size_t j) const 
    {
        if(i >= _shape.first || j >= _shape.second)
        { 
            MGCPP_THROW_OUT_OF_RANGE("index out of range");
        }

        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        {
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        T* from = (_data + (i * _shape.second + j));
        T to;
        auto result = cuda_memcpy(
            &to, from, 1, cuda_memcpy_kind::device_to_host);

        if(!result)
        { 
            MGCPP_THROW_SYSTEM_ERROR(result.error());
        }

        return to;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    device_matrix<T, DeviceId, SO>&
    device_matrix<T, DeviceId, SO>::
    operator=(host_matrix<T, SO> const& cpu_mat)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { 
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        if(!_released)
        {
            (void)cuda_free(_data); 
            _released = true;
        }

        auto shape = cpu_mat.shape();
        _shape.first = shape.first;
        _shape.second = shape.second;

        size_t total_size = _shape.first * _shape.second;

        auto alloc_result = cuda_malloc<T>(total_size);
        if(!alloc_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
        }
        _released = false;
        
        auto cpy_result =
            cuda_memcpy(alloc_result.value(),
                        cpu_mat.data(),
                        total_size,
                        cuda_memcpy_kind::host_to_device);
        if(!cpy_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }
        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    inline T const*
    device_matrix<T, DeviceId, SO>::
    data() const noexcept
    {
        return _data;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    inline T*
    device_matrix<T, DeviceId, SO>::
    data_mutable() noexcept
    {
        return _data;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    inline T*
    device_matrix<T, DeviceId, SO>::
    release_data()
    {
        _released = true;
        return _data;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    inline thread_context*
    device_matrix<T, DeviceId, SO>::
    context() const noexcept
    {
        return _context;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    matrix_shape const&
    device_matrix<T, DeviceId, SO>::
    shape() const noexcept
    {
        return _shape;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO>
    device_matrix<T, DeviceId, SO>::
    ~device_matrix() noexcept
    {
        (void)cuda_set_device(DeviceId);

        if(!_released)
            (void)cuda_free(_data);
        global_context::reference_cnt_decr();
    }
}
