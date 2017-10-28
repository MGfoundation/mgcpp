
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
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>::
    device_matrix() noexcept
    : _context(&global_context::get_thread_context()),
        _shape(0, 0),
        _data(nullptr){}

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>::
    device_matrix(size_t i, size_t j)
        :_context(&global_context::get_thread_context()),
         _shape(i, j),
         _data(Alloc::allocate(_shape.first * _shape.second, DeviceId))
    {}

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>::
    device_matrix(size_t i, size_t j, T init)
        : _context(&global_context::get_thread_context()),
          _shape(i, j),
          _data(Alloc::allocate(_shape.first * _shape.second, DeviceId))
    {
        auto total_size = _shape.first * _shape.second;
        T* buffer = (T*)malloc(sizeof(T) * total_size);
        if(!buffer)
        {
            MGCPP_THROW_BAD_ALLOC;
        }

        std::fill(buffer, buffer + total_size, init);

        try
        {
            Alloc::copy_from_host(_data, buffer,
                                  total_size, DeviceId);
            free(buffer);
        }
        catch(std::system_error const& err)
        {
            free(buffer);
            (void)cuda_free(_data); 
            MGCPP_THROW_SYSTEM_ERROR(err);
        }
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>::
    device_matrix(device_matrix<T, DeviceId, SO, Alloc> const& other)
        :_context(&global_context::get_thread_context()),
         _shape(other._shape),
         _data(Alloc::allocate(_shape.first * _shape.second, DeviceId))
    {
        auto cpy_result =
            cuda_memcpy(_data,
                        other._data,
                        _shape.first * _shape.second,
                        cuda_memcpy_kind::device_to_device);

        if(!cpy_result)
        {
            Alloc::deallocate(_data);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>::
    device_matrix(host_matrix<T, SO> const& cpu_mat)
        : _context(&global_context::get_thread_context()),
          _shape(cpu_mat.shape()),
          _data(Alloc::allocate(_shape.first * _shape.second, DeviceId))
    {
        Alloc::copy_from_host(_data, cpu_mat.data(), DeviceId);
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>::
    device_matrix(device_matrix<T, DeviceId, SO, Alloc>&& other) noexcept
        : _context(&global_context::get_thread_context()),
          _shape(std::move(other._shape)),
          _data(other._data)
    {
        other._data = nullptr;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>&
    device_matrix<T, DeviceId, SO, Alloc>::
    operator=(device_matrix<T, DeviceId, SO, Alloc> const& other)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { 
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        if(!_data)
        {
            Alloc::deallocate(_data);
            _data = nullptr;
        }

        auto total_size = other._shape.first * other._shape.second;
        _data = Alloc::allocate(total_size);
        _shape = other._shape;

        auto cpy_result =
            cuda_memcpy(_data,
                        other._data,
                        total_size,
                        cuda_memcpy_kind::device_to_device);
        if(!cpy_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>&
    device_matrix<T, DeviceId, SO, Alloc>::
    operator=(device_matrix<T, DeviceId, SO, Alloc>&& other) noexcept
    {
        if(!_data)
        { 
            (void)cuda_set_device(DeviceId);
            (void)cuda_free(_data);
            _data = nullptr;
        }
        _data = other._data;
        other._data = nullptr;
        _shape = std::move(other._shape);

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>&
    device_matrix<T, DeviceId, SO, Alloc>::
    resize(size_t i, size_t j)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { 
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        if(!_data)
        {
            Alloc::deallocate(_data);
            _data = nullptr;
        }

        _data = Alloc::allocate(i * j);
        _shape = std::make_pair(i, j);

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>&
    device_matrix<T, DeviceId, SO, Alloc>::
    resize(size_t i, size_t j, T init)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { 
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        Alloc::deallocate(_data);
        _data= nullptr;

        size_t total_size = i * j;
        _data = Alloc::allocate(total_size);

        _shape = std::make_pair(i, j);

        T* buffer = (T*)malloc(sizeof(T) * total_size);
        if(!buffer)
        {
            MGCPP_THROW_BAD_ALLOC;
        }

        std::fill(buffer, buffer + total_size, init);

        try
        { Alloc::copy_from_host(_data, buffer, total_size); }
        catch(std::system_error const& err)
        {
            free(buffer);
            throw err;
        }
        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    host_matrix<T, SO>
    device_matrix<T, DeviceId, SO, Alloc>::
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

        try
        {
            Alloc::copy_to_host(host_memory, _data, total_size);
        }
        catch(std::system_error const& err)
        {
            free(host_memory);
            throw err;
        }

        return host_matrix<T, SO>(_shape.first,
                                  _shape.second,
                                  host_memory);
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>&
    device_matrix<T, DeviceId, SO, Alloc>::
    zero()
    {
        if(!_data)
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
             storage_order SO,
             typename Alloc>
    T
    device_matrix<T, DeviceId, SO, Alloc>::
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
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>&
    device_matrix<T, DeviceId, SO, Alloc>::
    operator=(host_matrix<T, SO> const& cpu_mat)
    {
        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { 
            MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
        }

        if(!_data)
        {
            Alloc::deallocate(_data);
            _data = nullptr;
        }

        auto shape = cpu_mat.shape();
        _shape.first = shape.first;
        _shape.second = shape.second;


        auto total_size = _shape.first * _shape.second;
        auto _data = Alloc::allocate(total_size);
        Alloc::copy_from_host(_data, cpu_mat.data(), total_size);
        
        return *this;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    inline T const*
    device_matrix<T, DeviceId, SO, Alloc>::
    data() const noexcept
    {
        return _data;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    inline T*
    device_matrix<T, DeviceId, SO, Alloc>::
    data_mutable() noexcept
    {
        return _data;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    inline T*
    device_matrix<T, DeviceId, SO, Alloc>::
    release_data() noexcept
    {
        T* temp = _data;
        _data = nullptr;
        return temp;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    inline thread_context*
    device_matrix<T, DeviceId, SO, Alloc>::
    context() const noexcept
    {
        return _context;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    matrix_shape const&
    device_matrix<T, DeviceId, SO, Alloc>::
    shape() const noexcept
    {
        return _shape;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>::
    ~device_matrix() noexcept
    {
        (void)cuda_set_device(DeviceId);

        if(_data)
            (void)cuda_free(_data);
        global_context::reference_cnt_decr();
    }
}
