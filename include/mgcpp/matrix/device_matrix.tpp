
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/kernels/mgblas_helpers.hpp>
#include <mgcpp/system/exception.hpp>

#include <type_traits>
#include <iostream>

namespace mgcpp
{
    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    device_matrix<Type, DeviceId, Alloc>::
    device_matrix() noexcept
    : _context(&global_context::get_thread_context()),
        _shape(0, 0),
        _allocator(),
        _data(nullptr),
        _capacity(0) {}

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    device_matrix<Type, DeviceId, Alloc>::
    device_matrix(Alloc const& alloc) 
        : _context(&global_context::get_thread_context()),
          _shape(0, 0),
          _allocator(alloc),
          _data(nullptr),
          _capacity(0) {}

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    device_matrix<Type, DeviceId, Alloc>::
    device_matrix(size_t i, size_t j, Alloc const& alloc)
        :_context(&global_context::get_thread_context()),
         _shape(i, j),
         _allocator(alloc),
         _data(_allocator.device_allocate(_shape.first * _shape.second)),
         _capacity(i * j) {}

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    device_matrix<Type, DeviceId, Alloc>::
    device_matrix(size_t i, size_t j, value_type init, Alloc const& alloc)
        : _context(&global_context::get_thread_context()),
          _shape(i, j),
          _allocator(alloc),
          _data(_allocator.device_allocate(_shape.first * _shape.second)),
          _capacity(i * j)
    {
        size_t total_size = _shape.first * _shape.second;

        auto dinit = mgcpp_cast<device_pointer>(&init);
        auto status = mgblas_fill(_data,
                                  *dinit,
                                  total_size);
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }
    }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    device_matrix<Type, DeviceId, Alloc>::
    device_matrix(size_t i, size_t j, const_pointer data, Alloc const& alloc)
        : _context(&global_context::get_thread_context()),
          _shape(i, j),
          _allocator(alloc),
          _data(_allocator.device_allocate(_shape.first * _shape.second)),
          _capacity(i * j)
    {
        size_t total_size = _shape.first * _shape.second;

        try
        { _allocator.copy_from_host(_data, data, total_size); }
        catch(std::system_error const& err)
        {
            _allocator.device_deallocate(_data, total_size);
            MGCPP_THROW_SYSTEM_ERROR(err);
        }
    }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    size_t
    device_matrix<Type, DeviceId, Alloc>::
    determine_ndim(std::initializer_list<
                       std::initializer_list<value_type>> const& list) const noexcept
    {
        auto max_elem = std::max(list.begin(),
                                 list.end(),
                                 [](auto const& first,
                                    auto const& second)
                                 { return first->size() < second->size(); });

        if(max_elem == list.end())
            return list.begin()->size();
        else
            return max_elem->size();
    }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    device_matrix<Type, DeviceId, Alloc>::
    device_matrix(std::initializer_list<
                      std::initializer_list<value_type>> const& init_list,
                  Alloc const& alloc)
        : _context(&global_context::get_thread_context()),
          _shape(init_list.size(), determine_ndim(init_list)),
          _allocator(alloc),
          _data(_allocator.device_allocate(_shape.first * _shape.second)),
          _capacity(_shape.first * _shape.second)
    {
        size_t total_size = _shape.first * _shape.second;

        pointer buffer = _allocator.allocate(total_size);

        size_t i = 0;
        for(const auto& row_list : init_list)
        {
            // std::fill(std::copy(row_list.begin(),
            //                     row_list.end(),
            //                     buffer + i * _shape.second ),
            //           buffer + (i + 1) * _shape.second,
            //           Type());
            // ++i;
            size_t j = 0;
            for(Type elem : row_list)
            {
                buffer[i + _shape.first * j] = elem;
                ++j;
            }
            ++i;
        }

        try
        {
            _allocator.copy_from_host(_data, buffer, total_size);
            _allocator.deallocate(buffer, total_size);
        }
        catch(std::system_error const& err)
        {
            _allocator.deallocate(buffer, total_size);
            _allocator.device_deallocate(_data, _capacity);
            MGCPP_THROW(err);
        }
    }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    template<typename HostMat, typename>
    device_matrix<Type, DeviceId, Alloc>::
    device_matrix(HostMat const& host_mat, Alloc const& alloc)
        :_context(&global_context::get_thread_context()),
         _shape(0, 0),
         _allocator(alloc),
         _data(nullptr),
         _capacity(0)
    {
        adapter<HostMat> adapt{};

        pointer host_p;
        adapt(host_mat, &host_p, &_shape.first, &_shape.second);

        size_t total_size = _shape.first * _shape.second;
        _data = _allocator.device_allocate(total_size);
        _capacity = total_size;
        _allocator.copy_from_host(_data, host_p, total_size);
    }


    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    device_matrix<Type, DeviceId, Alloc>::
    device_matrix(device_matrix<Type, DeviceId, Alloc> const& other)
        :_context(&global_context::get_thread_context()),
         _shape(other._shape),
         _allocator(),
         _data(_allocator.device_allocate(_shape.first * _shape.second)),
         _capacity(_shape.first * _shape.second)
    {
        auto cpy_result = cuda_memcpy(_data, other._data,
                                      _shape.first * _shape.second,
                                      cuda_memcpy_kind::device_to_device);

        if(!cpy_result)
        {
            _allocator.device_deallocate(_data, _capacity);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }
    }


    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    template<typename DenseMatrix>
    device_matrix<Type, DeviceId, Alloc>::
    device_matrix(dense_matrix<DenseMatrix, Type, DeviceId> const& other)
        :_context(&global_context::get_thread_context()),
         _shape((~other)._shape),
         _allocator(),
         _data(_allocator.device_allocate(_shape.first * _shape.second)),
         _capacity(_shape.first * _shape.second)
    {
        auto cpy_result = cuda_memcpy(_data, (~other)._data,
                                      _shape.first * _shape.second,
                                      cuda_memcpy_kind::device_to_device);

        if(!cpy_result)
        {
            _allocator.device_deallocate(_data, _capacity);
            MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
        }
    }


    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    device_matrix<Type, DeviceId, Alloc>::
    device_matrix(device_matrix<Type, DeviceId, Alloc>&& other) noexcept
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
             size_t DeviceId,
             typename Alloc>
    device_matrix<Type, DeviceId, Alloc>&
    device_matrix<Type, DeviceId, Alloc>::
    operator=(device_matrix<Type, DeviceId, Alloc> const& other)
    {
        auto shape = other._shape;
        size_t other_size = shape.first * shape.second;
        if(other_size > _capacity)
        {
            if(_data)
            {
                _allocator.device_deallocate(_data, _capacity);
                _capacity = 0;
            }
            _data = _allocator.device_allocate(other_size); 
            _capacity = other_size;
        }
        auto cpy_result = cuda_memcpy(_data,
                                      other._data,
                                      other_size,
                                      cuda_memcpy_kind::device_to_device);
        _shape = other._shape;

        if(!cpy_result)
        { MGCPP_THROW_SYSTEM_ERROR(cpy_result.error()); }

        return *this;
    }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    template<typename DenseMatrix>
    device_matrix<Type, DeviceId, Alloc>&
    device_matrix<Type, DeviceId, Alloc>::
    operator=(dense_matrix<DenseMatrix, Type, DeviceId> const& other)
    {
        auto const& other_densemat = ~other;

        auto shape = other_densemat._shape;
        size_t other_size = shape.first * shape.second;
        if(other_size > _capacity)
        {
            if(_data)
            {
                _allocator.device_deallocate(_data, _capacity);
                _capacity = 0;
            }
            _data = _allocator.device_allocate(other_size); 
            _capacity = other_size;
        }
        auto cpy_result = cuda_memcpy(_data,
                                      other_densemat._data,
                                      other_size,
                                      cuda_memcpy_kind::device_to_device);
        _shape = other_densemat._shape;

        if(!cpy_result)
        { MGCPP_THROW_SYSTEM_ERROR(cpy_result.error()); }

        return *this;
    }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    device_matrix<Type, DeviceId, Alloc>&
    device_matrix<Type, DeviceId, Alloc>::
    operator=(device_matrix<Type, DeviceId, Alloc>&& other) noexcept
    {
        if(_data)
        { 
            try
            {
                _allocator.device_deallocate( _data, _capacity);
            } catch(...){};
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
             size_t DeviceId,
             typename Alloc>
    device_matrix<Type, DeviceId, Alloc>&
    device_matrix<Type, DeviceId, Alloc>::
    resize(size_t i, size_t j)
    {
        size_t total_size = i * j;
        if(total_size > _capacity)
        {
            if(_data)
            {
                _allocator.device_deallocate(_data, _capacity);
                _capacity = 0;
            }
            _data = _allocator.device_allocate(total_size); 
            _capacity = total_size;
        }

        _shape = std::make_pair(i, j);

        return *this;
    }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    device_matrix<Type, DeviceId, Alloc>&
    device_matrix<Type, DeviceId, Alloc>::
    resize(size_t i, size_t j, value_type init)
    {
        size_t total_size = i * j;
        if(total_size > _capacity)
        {
            if(_data)
            {
                _allocator.device_deallocate(_data, _capacity);
                _capacity = 0;
            }
            _data = _allocator.device_allocate(total_size); 
            _capacity = total_size;
        }

        _shape = std::make_pair(i, j);

        auto status = mgblas_fill(_data, init, total_size);

        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return *this;
    }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    device_matrix<Type, DeviceId, Alloc>&
    device_matrix<Type, DeviceId, Alloc>::
    zero()
    {
        if(!_data)
        { MGCPP_THROW_RUNTIME_ERROR("gpu memory wasn't allocated"); }

        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error()); }

        auto set_result = cuda_memset(_data,
                                      static_cast<Type>(0),
                                      _shape.first * _shape.second);
        if(!set_result)
        { MGCPP_THROW_SYSTEM_ERROR(set_result.error()); }

        return *this;
    }


    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    column_view<device_matrix<Type, DeviceId, Alloc>, Type, DeviceId>
    device_matrix<Type, DeviceId, Alloc>::
    column(size_t i) noexcept
    { return column_view<this_type, Type, DeviceId>(*this, i); }


    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    row_view<device_matrix<Type, DeviceId, Alloc>, Type, DeviceId>
    device_matrix<Type, DeviceId, Alloc>::
    row(size_t i) noexcept
    { return row_view<this_type, Type, DeviceId>(*this, i); }


    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    typename device_matrix<Type, DeviceId, Alloc>::value_type
    device_matrix<Type, DeviceId, Alloc>::
    check_value(size_t i, size_t j) const 
    {
        if(i >= _shape.first || j >= _shape.second)
        { MGCPP_THROW_OUT_OF_RANGE("index out of range."); }

        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error()); }

        device_pointer from = (_data + (i + _shape.first * j));
        value_type to;
        _allocator.copy_to_host(&to, from, 1);

        return to;
    }


    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    void
    device_matrix<Type, DeviceId, Alloc>::
    set_value(size_t i, size_t j, value_type value)
    {
        if(i >= _shape.first || j >= _shape.second)
        { MGCPP_THROW_OUT_OF_RANGE("index out of range."); }

        auto set_device_stat = cuda_set_device(DeviceId);
        if(!set_device_stat)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error()); }

        device_pointer to = (_data + (i + _shape.first * j));
        _allocator.copy_from_host(to, &value, 1);
    }


    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    void
    device_matrix<Type, DeviceId, Alloc>::
    copy_to_host(pointer host_p) const
    {
        size_t total_size = _shape.first * _shape.second;
        if(!host_p)
        { MGCPP_THROW_RUNTIME_ERROR("provided pointer is null"); }
        _allocator.copy_to_host(host_p, _data, total_size);
    }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    typename device_matrix<Type, DeviceId, Alloc>::const_device_pointer
    device_matrix<Type, DeviceId, Alloc>::
    data() const noexcept
    { return _data; }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    typename device_matrix<Type, DeviceId, Alloc>::device_pointer
    device_matrix<Type, DeviceId, Alloc>::
    data_mutable() noexcept
    { return _data; }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    typename device_matrix<Type, DeviceId, Alloc>::device_pointer
    device_matrix<Type, DeviceId, Alloc>::
    release_data() noexcept
    {
        device_pointer temp = _data;
        _data = nullptr;
        _capacity = 0;
        return temp;
    }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    size_t 
    device_matrix<Type, DeviceId, Alloc>::
    capacity() const noexcept
    { return _capacity; }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    thread_context*
    device_matrix<Type, DeviceId, Alloc>::
    context() const noexcept
    { return _context; }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    std::pair<size_t, size_t> const&
    device_matrix<Type, DeviceId, Alloc>::
    shape() const noexcept
    { return _shape; }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    Alloc&
    device_matrix<Type, DeviceId, Alloc>::
    allocator() noexcept
    { return _allocator; }

    template<typename Type,
             size_t DeviceId,
             typename Alloc>
    device_matrix<Type, DeviceId, Alloc>::
    ~device_matrix() noexcept
    {
        if(_data)
        {
            try{ _allocator.device_deallocate(_data, _capacity); }
            catch(...){};
        }
        global_context::reference_cnt_decr();
    }
}
