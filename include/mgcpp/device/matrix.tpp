
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
        _data(nullptr),
        _capacity(0) {}

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>::
    device_matrix(size_t i, size_t j)
        :_context(&global_context::get_thread_context()),
         _shape(i, j),
         _data(device_allocate(_shape.first * _shape.second)),
         _capacity(i * j) {}

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>::
    device_matrix(size_t i, size_t j, T init)
        : _context(&global_context::get_thread_context()),
          _shape(i, j),
          _data(device_allocate(_shape.first * _shape.second)),
          _capacity(i * j)
    {
        size_t total_size = _shape.first * _shape.second;

        T* buffer = allocate(total_size);
        std::fill(buffer, buffer + total_size, init);

        try
        {
            copy_from_host(_data, buffer, total_size);
            deallocate(buffer, total_size);
        }
        catch(std::system_error const& err)
        {
            deallocate(buffer, total_size);
            device_deallocate(_data, DeviceId);
            MGCPP_THROW_SYSTEM_ERROR(err);
        }
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>::
    device_matrix(size_t i, size_t j, T const* data)
        : _context(&global_context::get_thread_context()),
          _shape(i, j),
          _data(device_allocate(_shape.first * _shape.second)),
          _capacity(i * j)
    {
        try
        {
            copy_from_host(_data, data, _shape.first * _shape.second);
        }
        catch(std::system_error const& err)
        {
            device_deallocate(_data, _shape.first * _shape.second);
            MGCPP_THROW_SYSTEM_ERROR(err);
        }
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>::
    device_matrix(std::initializer_list<std::initializer_list<T>> const& array)
        : _context(&global_context::get_thread_context()),
          _shape(array.size(), std::max(array.begin(),
                                        array.end(),
                                        [](auto const& first, auto const& second)
                                        { return first.size() < second.size(); })),
          _data(device_allocate(_shape.first * _shape.second)),
          _capacity(_shape.first * _shape.second)
    {
        T* buffer = allocate(_shape.first * _shape.second);
        size_t i = 0;
        for(auto const& row : array)
        {
            std::fill(std::copy(row.begin(),
                                row.end(),
                                buffer + i * _shape.second),
                      buffer + (i+1) * _shape.second,
                      T());
            ++i; 
        }

        try
        {
            copy_from_host(_data, buffer, _shape.first * _shape.second);
            deallocate(buffer);
        }
        catch(std::system_error const& err)
        {
            deallocate(buffer);
            throw err;
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
         _data(device_allocate(_shape.first * _shape.second)),
         _capacity(_shape.first * _shape.second)
    {
        auto cpy_result = cuda_memcpy(_data, other._data,
                                      _shape.first * _shape.second,
                                      cuda_memcpy_kind::device_to_device);

        if(!cpy_result)
        {
            device_deallocate(_data, _capacity);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }
    }

    // template<typename T,
    //          size_t DeviceId,
    //          storage_order SO,
    //          typename Alloc>
    // device_matrix<T, DeviceId, SO, Alloc>::
    // device_matrix(host_matrix<T, SO> const& cpu_mat)
    //     : _context(&global_context::get_thread_context()),
    //       _shape(cpu_mat.shape()),
    //       _data(device_allocate(_shape.first * _shape.second)),
    //       _capacity(_shape.first * _shape.second)
    // {
    //     copy_from_host(_data, cpu_mat.data());
    // }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>::
    device_matrix(device_matrix<T, DeviceId, SO, Alloc>&& other) noexcept
        : _context(&global_context::get_thread_context()),
          _shape(std::move(other._shape)),
          _data(other._data),
          _capacity(other._capacity)
    {
        other._data = nullptr;
        other._capacity = 0;
    }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>&
    device_matrix<T, DeviceId, SO, Alloc>::
    operator=(device_matrix<T, DeviceId, SO, Alloc> const& other)
    {
        size_t other_size = other._shape.first * other._shape.second;
        if(other_size > _capacity)
        {
            device_deallocate(_data, _capacity);
            _data = device_allocate(other_size); 
            _capacity = other_size;
            _shape = other._shape;
        }

        auto cpy_result = cuda_memcpy(_data,
                                      other._data,
                                      other_size,
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
        if(_data)
        { 
            try { device_deallocate(
                    _data, _shape.first * _shape.second); } catch(...){};
            _data = nullptr;
        }
        _data = other._data;
        other._data = nullptr;
        _capacity = other._capacity;
        other._capacity = 0;
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
        if(_data)
        {
            device_deallocate(_data, _shape.first * _shape.second);
            _data = nullptr;
        }

        _data = device_allocate(i * j);
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
        if(_data)
        {
            device_deallocate(_data, _shape.first * _shape.second);
            _data = nullptr;
        }

        size_t total_size = i * j;
        _data = device_allocate(total_size);
        _shape = std::make_pair(i, j);

        T* buffer = allocate(total_size);
        std::fill(buffer, buffer + total_size, init);

        try
        { copy_from_host(_data, buffer, total_size); }
        catch(std::system_error const& err)
        {
            deallocate(buffer, total_size);
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
    copy_to_host() 
    {
        size_t total_size = _shape.first * _shape.second;
        T* buffer = allocate(total_size);

        try
        { copy_to_host(buffer, _data, total_size); }
        catch(std::system_error const& err)
        {
            deallocate(buffer, total_size);
            throw err;
        }

        return host_matrix<T, SO>(_shape.first,
                                  _shape.second,
                                  buffer);
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

        auto set_result = cuda_memset(_data,
                                      static_cast<T>(0),
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

    // template<typename T,
    //          size_t DeviceId,
    //          storage_order SO,
    //          typename Alloc>
    // device_matrix<T, DeviceId, SO, Alloc>&
    // device_matrix<T, DeviceId, SO, Alloc>::
    // operator=(host_matrix<T, SO> const& cpu_mat)
    // {
    //     auto total_size = _shape.first * _shape.second;
    //     if(_data)
    //     {
    //         device_deallocate(_data, total_size);
    //         _data = nullptr;
    //     }

    //     auto shape = cpu_mat.shape();
    //     _shape.first = shape.first;
    //     _shape.second = shape.second;

    //     auto _data = device_allocate(total_size);
    //     copy_from_host(_data, cpu_mat.data(), total_size);
        
    //     return *this;
    // }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    inline T const*
    device_matrix<T, DeviceId, SO, Alloc>::
    data() const noexcept
    { return _data; }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    inline T*
    device_matrix<T, DeviceId, SO, Alloc>::
    data_mutable() noexcept
    { return _data; }

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
    inline size_t 
    device_matrix<T, DeviceId, SO, Alloc>::
    capacity() const noexcept
    { return _capacity; }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    inline thread_context*
    device_matrix<T, DeviceId, SO, Alloc>::
    context() const noexcept
    { return _context; }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    std::pair<size_t, size_t> const&
    device_matrix<T, DeviceId, SO, Alloc>::
    shape() const noexcept
    { return _shape; }

    template<typename T,
             size_t DeviceId,
             storage_order SO,
             typename Alloc>
    device_matrix<T, DeviceId, SO, Alloc>::
    ~device_matrix() noexcept
    {
        if(_data)
            try{
                device_deallocate(_data, _capacity);
            }catch(...){};
        global_context::reference_cnt_decr();
    }
}
