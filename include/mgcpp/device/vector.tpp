
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/global_context.hpp>
#include <mgcpp/context/thread_context.hpp>
//#include <mgcpp/host/vector.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/device/vector.hpp>
#include <mgcpp/system/exception.hpp>

#include <algorithm>

namespace mgcpp
{
    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    device_vector<T, DeviceId, Allign, Alloc>::
    device_vector() noexcept
    :_context(&global_context::get_thread_context()),
        _shape(0),
        _data(nullptr),
        _capacity(0) {}

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    device_vector<T, DeviceId, Allign, Alloc>::
    device_vector(size_t size)
        :_context(&global_context::get_thread_context()),
         _shape(size),
         _data(device_allocate(_shape)),
         _capacity(_shape) {}

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    device_vector<T, DeviceId, Allign, Alloc>::
    device_vector(size_t size, T init)
        : _context(&global_context::get_thread_context()),
          _shape(size),
          _data(device_allocate(_shape)),
          _capacity(_shape)
    {
        T* buffer = allocate(_shape);
        std::fill(buffer, buffer + _shape, init);

        try
        {
            copy_from_host(_data, buffer, _shape);
            deallocate(buffer, _shape);
        }
        catch(std::system_error const& err)
        {
            deallocate(buffer, _shape);
            device_deallocate(_data, _capacity);
            MGCPP_THROW_SYSTEM_ERROR(err);
        }
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    device_vector<T, DeviceId, Allign, Alloc>::
    device_vector(size_t size, T const* data)
        : _context(&global_context::get_thread_context()),
          _shape(size),
          _data(device_allocate(_shape)),
          _capacity(size)
    {
        try
        { copy_from_host(_data, data, _shape); }
        catch(std::system_error const& err)
        {
            device_deallocate(_data, _capacity);
            MGCPP_THROW_SYSTEM_ERROR(err);
        }
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    device_vector<T, DeviceId, Allign, Alloc>::
    device_vector(std::initializer_list<T> const& array)
        : _context(&global_context::get_thread_context()),
          _shape(array.size()),
          _data(device_allocate(_shape)),
          _capacity(_shape)
    {
        T* buffer = allocate(_shape);
        std::copy(array.begin(), array.end(), buffer);

        try
        {
            copy_from_host(_data, buffer, _shape);
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
             allignment Allign,
             typename Alloc>
    device_vector<T, DeviceId, Allign, Alloc>::
    device_vector(device_vector<T, DeviceId, Allign, Alloc> const& other)
        :_context(&global_context::get_thread_context()),
         _shape(other._shape),
         _data(device_allocate(_shape)),
         _capacity(_shape)
    {
        auto cpy_result = cuda_memcpy(_data, other._data, _shape,
                                      cuda_memcpy_kind::device_to_device);

        if(!cpy_result)
        {
            device_deallocate(_data, _shape);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }
    }

    
    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    device_vector<T, DeviceId, Allign, Alloc>::
    device_vector(device_vector<T, DeviceId, Allign, Alloc>&& other) noexcept
        : _context(&global_context::get_thread_context()),
          _shape(std::move(other._shape)),
          _data(other._data),
          _capacity(other._capacity)
    {
        other._data = nullptr;
        other._capacity = 0;
    }

    // template<typename T,
    //          size_t DeviceId,
    //          allignment Allign,
    //          typename Alloc>
    // device_vector<T, DeviceId, Allign, Alloc>::
    // device_vector(host_vector<T, Allign, Alloc> const& other)
    //     :_data(nullptr),
    //      _context(&global_context::get_thread_context()),
    //      _size(other._size),
    //      _released(true)
    // {
    //     auto set_device_stat = cuda_set_device(DeviceId);
    //     if(!set_device_stat)
    //     {
    //         MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
    //     }

    //     auto alloc_result = cuda_malloc<T>(other._size);
    //     if(!alloc_result)
    //     {
    //         MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
    //     }
    //     _released = false;
    //     _size = other._size;

    //     auto cpy_result =
    //         cuda_memcpy(alloc_result.value(),
    //                     other._data,
    //                     _size,
    //                     cuda_memcpy_kind::host_to_device);

    //     if(!cpy_result)
    //     {
    //         (void)cuda_free(alloc_result.value());
    //         MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
    //     }

    //     _data = alloc_result.value();
    // }

    template<typename T,
             size_t DeviceId,
             allignment Allign, typename Alloc>
    device_vector<T, DeviceId, Allign, Alloc>&
    device_vector<T, DeviceId, Allign, Alloc>::
    operator=(device_vector<T, DeviceId, Allign, Alloc> const& other)
    {
        if(other._shape > _capacity)
        {
            device_deallocate(_data, _capacity);
            _data = device_allocate(other._shape); 
            _capacity = other._shape;
            _shape = other._shape;
        }

        auto cpy_result = cuda_memcpy(_data,
                                      other._data,
                                      other._shape,
                                      cuda_memcpy_kind::device_to_device);
        if(!cpy_result)
        {
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    device_vector<T, DeviceId, Allign, Alloc>&
    device_vector<T, DeviceId, Allign, Alloc>::
    operator=(device_vector<T, DeviceId, Allign, Alloc>&& other) noexcept
    {
        if(_data)
        { 
            try { device_deallocate(_data, _shape); } catch(...){};
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
             allignment Allign,
             typename Alloc>
    device_vector<T, DeviceId, Allign, Alloc>&
    device_vector<T, DeviceId, Allign, Alloc>::
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

        auto set_result = cuda_memset(_data, static_cast<T>(0), _shape);
        if(!set_result)
        { 
            MGCPP_THROW_SYSTEM_ERROR(set_result.error());
        }

        return *this;
    }

    // template<typename T,
    //          size_t DeviceId,
    //          allignment Allign,
    //          typename Alloc>
    // device_vector<T, DeviceId, Allign, Alloc>&
    // device_vector<T, DeviceId, Allign, Alloc>::
    // operator=(host_vector<T, Allign, Alloc> const& host) 
    // {
    //     auto set_device_stat = cuda_set_device(DeviceId);
    //     if(!set_device_stat)
    //     {
    //         MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
    //     }

    //     if(_released)
    //     {
    //         auto free_stat = cuda_free(_data);
    //         if(!free_stat)
    //         {
    //             MGCPP_THROW_SYSTEM_ERROR(free_stat.error());
    //         }
    //         _released = true;
    //     }

    //     auto alloc_result = cuda_malloc<T>(host._size);
    //     if(!alloc_result)
    //     {
    //         MGCPP_THROW_SYSTEM_ERROR(alloc_result.error());
    //     }
    //     _released = true;

    //     auto cpy_result =
    //         cuda_memcpy(alloc_result, host.data(), _size,
    //                     cuda_memcpy_kind::host_to_device);

    //     _data = alloc_result;

    //     if(!cpy_result)
    //     {
    //         MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
    //     }

    //     return *this;
    // }

    // template<typename T,
    //          size_t DeviceId,
    //          allignment Allign,
    //          typename Alloc>
    // host_vector<T, Allign, Alloc> 
    // device_vector<T, DeviceId, Allign, Alloc>::
    // copy_to_host() const
    // {
    //     auto set_device_stat = cuda_set_device(DeviceId);
    //     if(!set_device_stat)
    //     {
    //         MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
    //     }

    //     T* host_memory = (T*)malloc(_size * sizeof(T));
    //     if(!host_memory)
    //     {
    //         MGCPP_THROW_BAD_ALLOC;
    //     }
        
    //     auto cpy_result =
    //         cuda_memcpy(host_memory, _data, _size,
    //                     cuda_memcpy_kind::device_to_host);
    //     if(!cpy_result)
    //     {
    //         free(host_memory);
    //         MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
    //     }

    //     return host_vector<T, Allign, typename Alloc>(_size, host_memory);
    // }


    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    T
    device_vector<T, DeviceId, Allign, Alloc>::
    check_value(size_t i) const
    {
        if(i >= _shape)
        {
            MGCPP_THROW_OUT_OF_RANGE("index out of range");
        }

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
        {
            MGCPP_THROW_SYSTEM_ERROR(result.error());
        }

        return to;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    inline T*
    device_vector<T, DeviceId, Allign, Alloc>::
    release_data() noexcept
    {
        T* temp = _data;
        _data = nullptr;
        return temp;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    inline thread_context*
    device_vector<T, DeviceId, Allign, Alloc>::
    context() const noexcept
    { return _context; }

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    size_t 
    device_vector<T, DeviceId, Allign, Alloc>::
    capacity() const noexcept
    { return _capacity; }

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    T const*
    device_vector<T, DeviceId, Allign, Alloc>::
    data() const noexcept
    { return _data; }

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    T*
    device_vector<T, DeviceId, Allign, Alloc>::
    data_mutable() noexcept
    { return _data; }

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    size_t
    device_vector<T, DeviceId, Allign, Alloc>::
    shape() const noexcept
    { return _shape; }

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    size_t
    device_vector<T, DeviceId, Allign, Alloc>::
    size() const noexcept
    { return _shape; }
    
    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    device_vector<T, DeviceId, Allign, Alloc>::
    ~device_vector() noexcept
    {
        if(!_data)
            try{
                device_deallocate(_data, _capacity);
            }catch(...){};
        global_context::reference_cnt_decr();
    }
}
