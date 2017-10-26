
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/host/matrix.hpp>
#include <mgcpp/system/exception.hpp>

#include <algorithm>

namespace mgcpp
{
    template<typename T,
             storage_order SO>
    host_matrix<T, SO>::
    host_matrix() noexcept
    : _data(nullptr),
        _m_dim(0),
        _n_dim(0) {}


    template<typename T,
             storage_order SO>
    host_matrix<T, SO>::
    host_matrix(size_t i, size_t j)
        : _data(nullptr),
          _m_dim(i),
          _n_dim(j)
    {
        size_t total_size = i * j;
        T* ptr =
            (T*)malloc(sizeof(T) * total_size);

        if(!ptr)
            MGCPP_THROW_BAD_ALLOC;
        
        _data = ptr;
    }

    template<typename T,
             storage_order SO>
    host_matrix<T, SO>::
    host_matrix(size_t i, size_t j, T* data) noexcept
        : _data(data),
          _m_dim(i),
          _n_dim(j) {}

    template<typename T,
             storage_order SO>
    host_matrix<T, SO>::
    host_matrix(size_t i, size_t j, T init)
        : _data(nullptr),
          _m_dim(i),
          _n_dim(j)
    {
        size_t total_size = i * j;
        T* ptr = (T*)malloc(sizeof(T) * total_size);

        if(!ptr)
        {
            MGCPP_THROW_BAD_ALLOC;
        }

        std::fill(ptr, ptr + total_size, init);
        
        _data = ptr;
    }

    template<typename T,
             storage_order SO>
    inline T
    host_matrix<T, SO>::
    at(size_t i, size_t j) const
    {
        if(i >= _m_dim || j >= _n_dim)
            MGCPP_THROW_OUT_OF_RANGE("index out of range");

        return _data[i * _n_dim + j];
    }

    template<typename T,
             storage_order SO>
    inline T&
    host_matrix<T, SO>::
    at(size_t i, size_t j) 
    {
        if(i >= _m_dim || j >= _n_dim)
            MGCPP_THROW_OUT_OF_RANGE("index out of range");

        return _data[i * _n_dim + j];
    }

    template<typename T,
             storage_order SO>
    inline T
    host_matrix<T, SO>::
    operator()(size_t i, size_t j) const noexcept
    {
        return _data[i * _n_dim + j];
    }

    template<typename T,
             storage_order SO>
    inline T&
    host_matrix<T, SO>::
    operator()(size_t i, size_t j) noexcept
    {
        return _data[i * _n_dim + j];
    }

    template<typename T,
             storage_order SO>
    inline std::pair<size_t, size_t> 
    host_matrix<T, SO>::
    shape() const noexcept
    {
        return {_m_dim, _n_dim};
    }

    template<typename T,
             storage_order SO>
    inline T const*
    host_matrix<T, SO>::
    data() const
    {
        return _data;
    }

    template<typename T,
             storage_order SO>
    inline T*
    host_matrix<T, SO>::
    data_mutable() const
    {
        return _data;
    }

    template<typename T,
             storage_order SO>
    host_matrix<T, SO>::
    ~host_matrix() noexcept
    {
        (void)free(_data);
    }
}
