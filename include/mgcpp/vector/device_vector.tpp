
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/global_context.hpp>
#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/kernels/mgblas_helpers.hpp>
#include <mgcpp/vector/device_vector.hpp>
#include <mgcpp/system/exception.hpp>

#include <algorithm>

namespace mgcpp
{
    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    device_vector<Type, Align, DeviceId, Alloc>::
    device_vector() noexcept
    :_context(&global_context::get_thread_context()),
        _shape(0),
        _allocator(),
        _data(nullptr),
        _capacity(0) {}

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    device_vector<Type, Align, DeviceId, Alloc>::
    device_vector(Alloc const& alloc) noexcept
        :_context(&global_context::get_thread_context()),
         _shape(0),
         _allocator(alloc),
         _data(nullptr),
         _capacity(0) {}

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    device_vector<Type, Align, DeviceId, Alloc>::
    device_vector(size_t size, Alloc const& alloc)
        :_context(&global_context::get_thread_context()),
         _shape(size),
         _allocator(alloc),
         _data(_allocator.device_allocate(_shape)),
         _capacity(_shape) {}

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    device_vector<Type, Align, DeviceId, Alloc>::
    device_vector(size_t size, value_type init, Alloc const& alloc)
        : _context(&global_context::get_thread_context()),
          _shape(size),
          _allocator(alloc),
          _data(_allocator.device_allocate(_shape)),
          _capacity(_shape)
    {
        pointer buffer = _allocator.allocate(_shape);
        std::fill(buffer, buffer + _shape, init);

        auto dinit = mgcpp_cast<device_pointer>(&init);
        auto status = mgblas_fill(_data, *dinit, _shape);
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }
    }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    device_vector<Type, Align, DeviceId, Alloc>::
    device_vector(size_t size, const_pointer data, Alloc const& alloc)
        : _context(&global_context::get_thread_context()),
          _shape(size),
          _allocator(alloc),
          _data(_allocator.device_allocate(_shape)),
          _capacity(size)
    {
        try
        { _allocator.copy_from_host(_data, data, _shape); }
        catch(std::system_error const& err)
        {
            _allocator.device_deallocate(_data, _capacity);
            MGCPP_THROW_SYSTEM_ERROR(err);
        }
    }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    device_vector<Type, Align, DeviceId, Alloc>::
    device_vector(std::initializer_list<value_type> const& array,
                  Alloc const& alloc)
        : _context(&global_context::get_thread_context()),
          _shape(array.size()),
          _allocator(alloc),
          _data(_allocator.device_allocate(_shape)),
          _capacity(_shape)
    {
        try
        {
            // std::initializer_list's members are guaranteed to be
            // contiguous in memory: from C++11 § [support.initlist] 18.9/1
            _allocator.copy_from_host(_data, array.begin(), _shape);
        }
        catch(std::system_error const& err)
        {
            _allocator.device_deallocate(_data, _capacity);
            MGCPP_THROW_SYSTEM_ERROR(err);
        }
    }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    template<typename HostVec, typename>
    device_vector<Type, Align, DeviceId, Alloc>::
    device_vector(HostVec const& host_mat, Alloc const& alloc)
        :_context(&global_context::get_thread_context()),
         _shape(0),
         _allocator(alloc),
         _data(nullptr),
         _capacity(0)
    {
        adapter<HostVec> adapt{};

        pointer host_p; 
        adapt(host_mat, &host_p, &_shape);

        _capacity = _shape;
        _data = _allocator.device_allocate(_shape);
        _allocator.copy_from_host(_data, host_p, _shape);
    }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    device_vector<Type, Align, DeviceId, Alloc>::
    device_vector(device_vector<Type, Align, DeviceId, Alloc> const& other)
        :_context(&global_context::get_thread_context()),
         _shape(other._shape),
         _allocator(),
         _data(_allocator.device_allocate(_shape)),
         _capacity(_shape)
    {
        auto cpy_result = cuda_memcpy(_data, other._data, _shape,
                                      cuda_memcpy_kind::device_to_device);

        if(!cpy_result)
        {
            _allocator.device_deallocate(_data, _shape);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }
    }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    template<typename DenseVec>
    device_vector<Type, Align, DeviceId, Alloc>::
    device_vector(dense_vector<DenseVec, Type, Align, DeviceId> const& other)
        :_context(&global_context::get_thread_context()),
         _shape((~other)._shape),
         _allocator(),
         _data(_allocator.device_allocate(_shape)),
         _capacity(_shape)
    {
        auto cpy_result = cuda_memcpy(_data, (~other)._data, _shape,
                                      cuda_memcpy_kind::device_to_device);

        if(!cpy_result)
        {
            _allocator.device_deallocate(_data, _shape);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }
    }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    device_vector<Type, Align, DeviceId, Alloc>::
    device_vector(device_vector<Type, Align, DeviceId, Alloc>&& other) noexcept
        : _context(&global_context::get_thread_context()),
          _shape(std::move(other._shape)),
          _allocator(std::move(other._allocator)),
          _data(other._data),
          _capacity(other._capacity)
    {
        other._data = nullptr;
        other._capacity = 0;
    }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    template<typename DenseVec>
    device_vector<Type, Align, DeviceId, Alloc>&
    device_vector<Type, Align, DeviceId, Alloc>::
    operator=(dense_vector<DenseVec, Type, Align, DeviceId> const& other)
    {
        auto const& other_densevec = ~other;

        if(other_densevec._shape > _capacity)
        {
            if(_data)
            {
                _allocator.device_deallocate(_data, _capacity);
                _capacity = 0;
            }
            _data = _allocator.device_allocate(other_densevec._shape); 
            _capacity = other_densevec._shape;
        }

        auto cpy_result = cuda_memcpy(_data,
                                      other_densevec._data,
                                      other_densevec._shape,
                                      cuda_memcpy_kind::device_to_device);
        _shape = other_densevec._shape;

        if(!cpy_result)
        { MGCPP_THROW_SYSTEM_ERROR(cpy_result.error()); }

        return *this;
    }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    device_vector<Type, Align, DeviceId, Alloc>&
    device_vector<Type, Align, DeviceId, Alloc>::
    operator=(device_vector<Type, Align, DeviceId, Alloc> const& other)
    {
        if(other._shape > _capacity)
        {
            if(_data)
            {
                _allocator.device_deallocate(_data, _capacity);
                _capacity = 0;
            }
            _data = _allocator.device_allocate(other._shape); 
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

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    device_vector<Type, Align, DeviceId, Alloc>&
    device_vector<Type, Align, DeviceId, Alloc>::
    operator=(device_vector<Type, Align, DeviceId, Alloc>&& other) noexcept
    {
        if(_data)
        { 
            try { _allocator.device_deallocate(_data, _shape); } catch(...){};
            _data = nullptr;
        }
        _data = other._data;
        _capacity = other._capacity;
        _shape = std::move(other._shape);
        _allocator = std::move(other._allocator);
        other._data = nullptr;
        other._capacity = 0;

        return *this;
    }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    device_vector<Type, Align, DeviceId, Alloc>&
    device_vector<Type, Align, DeviceId, Alloc>::
    resize(size_t size)
    {
        if(size > _capacity)
        {
            if(_data)
            {
                _allocator.device_deallocate(_data, _capacity);
                _capacity = 0;
            }
            _data = _allocator.device_allocate(size); 
            _capacity = size;
        }
        _shape = size;

        return *this;
    }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    device_vector<Type, Align, DeviceId, Alloc>&
    device_vector<Type, Align, DeviceId, Alloc>::
    zero()
    {
        if(!_data)
        { MGCPP_THROW_RUNTIME_ERROR("gpu memory wasn't allocated"); }

        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error()); }

        auto set_result = cuda_memset(_data, static_cast<Type>(0), _shape);
        if(!set_result)
        { MGCPP_THROW_SYSTEM_ERROR(set_result.error()); }

        return *this;
    }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    void
    device_vector<Type, Align, DeviceId, Alloc>::
    copy_to_host(pointer host_p) const
    {
        if(!host_p)
        { MGCPP_THROW_RUNTIME_ERROR("provided pointer is null"); }
        _allocator.copy_to_host(host_p, _data, _shape);
    }


    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    typename device_vector<Type, Align, DeviceId, Alloc>::value_type
    device_vector<Type, Align, DeviceId, Alloc>::
    check_value(size_t i) const
    {
        if(i >= _shape)
        { MGCPP_THROW_OUT_OF_RANGE("index out of range."); }

        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error()); }

        device_pointer from = (_data + i);
        value_type to;
        _allocator.copy_to_host(&to, from, 1);

        return to;
    }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    void
    device_vector<Type, Align, DeviceId, Alloc>::
    set_value(size_t i, value_type value) 
    {
        if(i >= _shape)
        { MGCPP_THROW_OUT_OF_RANGE("index out of range."); }

        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error()); }

        device_pointer to = (_data + i);
        value_type from = value;
        _allocator.copy_from_host(to, &from, 1);
    }


    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    typename device_vector<Type, Align, DeviceId, Alloc>::device_pointer
    device_vector<Type, Align, DeviceId, Alloc>::
    release_data() noexcept
    {
        device_pointer temp = _data;
        _data = nullptr;
        _capacity = 0;
        return temp;
    }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    thread_context*
    device_vector<Type, Align, DeviceId, Alloc>::
    context() const noexcept
    { return _context; }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    size_t 
    device_vector<Type, Align, DeviceId, Alloc>::
    capacity() const noexcept
    { return _capacity; }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    typename device_vector<Type, Align, DeviceId, Alloc>::const_device_pointer
    device_vector<Type, Align, DeviceId, Alloc>::
    data() const noexcept
    { return _data; }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    typename device_vector<Type, Align, DeviceId, Alloc>::device_pointer
    device_vector<Type, Align, DeviceId, Alloc>::
    data_mutable() noexcept
    { return _data; }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    size_t
    device_vector<Type, Align, DeviceId, Alloc>::
    shape() const noexcept
    { return _shape; }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    size_t
    device_vector<Type, Align, DeviceId, Alloc>::
    size() const noexcept
    { return _shape; }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    Alloc& 
    device_vector<Type, Align, DeviceId, Alloc>::
    allocator() noexcept
    { return _allocator; }

    template<typename Type,
             alignment Align,
             size_t DeviceId,
             typename Alloc>
    device_vector<Type, Align, DeviceId, Alloc>::
    ~device_vector() noexcept
    {
        if(_data)
        {
            try
            { _allocator.device_deallocate(_data, _capacity); }
            catch(...){};
        }
        global_context::reference_cnt_decr();
    }
}
