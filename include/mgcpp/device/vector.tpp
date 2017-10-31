
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/global_context.hpp>
#include <mgcpp/context/thread_context.hpp>
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
            deallocate(buffer, _shape);
        }
        catch(std::system_error const& err)
        {
            deallocate(buffer, _shape);
            MGCPP_THROW(err);
        }
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    template<typename HostVec, typename>
    device_vector<T, DeviceId, Allign, Alloc>::
    device_vector(HostVec const& host_mat)
        :_context(&global_context::get_thread_context()),
         _shape(0),
         _data(nullptr),
         _capacity(0)
    {
        adapter<HostVec> adapt{};

        T* host_p;
        adapt(host_mat, &host_p, &_shape);

        _capacity = _shape;
        _data = device_allocate(_shape);
        copy_from_host(_data, host_p, _shape);
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    template<typename HostVec, typename Adapter, typename>
    device_vector<T, DeviceId, Allign, Alloc>::
    device_vector(HostVec const& host_vec, Adapter& adapter)
        :_context(&global_context::get_thread_context()),
         _shape(0),
         _data(nullptr),
         _capacity(0)
    {
        T* host_p;
        adapter(host_vec, &host_p, &_shape);

        _capacity = _shape;
        _data = device_allocate(_shape);
        copy_from_host(_data, host_p, _shape);
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
        }

        auto cpy_result = cuda_memcpy(_data,
                                      other._data,
                                      other._shape,
                                      cuda_memcpy_kind::device_to_device);
        _shape = other._shape;

        if(!cpy_result)
        { MGCPP_THROW_SYSTEM_ERROR(cpy_result.error()); }

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
        _capacity = other._capacity;
        _shape = std::move(other._shape);
        other._data = nullptr;
        other._capacity = 0;

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    device_vector<T, DeviceId, Allign, Alloc>&
    device_vector<T, DeviceId, Allign, Alloc>::
    resize(size_t size)
    {
        if(size > _capacity)
        {
            device_deallocate(_data, _capacity);
            _data = device_allocate(size); 
            _capacity = size;
        }
        _data = device_allocate(size);
        _shape = size;

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
        { MGCPP_THROW_RUNTIME_ERROR("gpu memory wasn't allocated"); }

        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error()); }

        auto set_result = cuda_memset(_data, static_cast<T>(0), _shape);
        if(!set_result)
        { MGCPP_THROW_SYSTEM_ERROR(set_result.error()); }

        return *this;
    }

    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    void
    device_vector<T, DeviceId, Allign, Alloc>::
    copy_to_host(T* host_p) const
    {
        if(!host_p)
        { MGCPP_THROW_RUNTIME_ERROR("provided pointer is null"); }
        copy_to_host(host_p, _data, _shape);
    }


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
        { MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error()); }

        T* from = (_data + i);
        T to;
        copy_to_host(&to, from, 1);

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
        if(_data)
            try{
                device_deallocate(_data, _capacity);
            }catch(...){};
        global_context::reference_cnt_decr();
    }
}
