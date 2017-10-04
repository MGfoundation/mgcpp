
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cpu/matrix.hpp>
#include <mgcpp/system/exception.hpp>

#include <algorithm>

namespace mgcpp
{
    template<typename ElemType,
             storage_order SO>
    cpu::matrix<ElemType, SO>::
    matrix() noexcept
    : _data(nullptr),
        _m_dim(0),
        _n_dim(0) {}


    template<typename ElemType,
             storage_order SO>
    cpu::matrix<ElemType, SO>::
    matrix(size_t i, size_t j)
        : _data(nullptr),
          _m_dim(i),
          _n_dim(j)
    {
        size_t total_size = i * j;
        ElemType* ptr =
            (ElemType*)malloc(sizeof(ElemType) * total_size);

        if(!ptr)
            MGCPP_THROW_BAD_ALLOC;
        
        _data = ptr;
    }

    template<typename ElemType,
             storage_order SO>
    cpu::matrix<ElemType, SO>::
    matrix(size_t i, size_t j, ElemType* data) noexcept
        : _data(data),
          _m_dim(i),
          _n_dim(j) {}

    template<typename ElemType,
             storage_order SO>
    cpu::matrix<ElemType, SO>::
    matrix(size_t i, size_t j, ElemType init)
        : _data(nullptr),
          _m_dim(i),
          _n_dim(j)
    {
        size_t total_size = i * j;
        ElemType* ptr =
            (ElemType*)malloc(sizeof(ElemType) * total_size);

        if(!ptr)
            MGCPP_THROW_BAD_ALLOC;

        std::fill(ptr, ptr + total_size, init);
        
        _data = ptr;
    }

    template<typename ElemType,
             storage_order SO>
    inline ElemType
    cpu::matrix<ElemType, SO>::
    operator()(size_t i, size_t j) const
    {
        if(i > _m_dim || j > _n_dim)
            MGCPP_THROW_OUT_OF_RANGE("index out of range");

        return _data[i * _n_dim + j];
    }

    template<typename ElemType,
             storage_order SO>
    inline ElemType&
    cpu::matrix<ElemType, SO>::
    operator()(size_t i, size_t j)
    {
        if(i > _m_dim || j > _n_dim)
            MGCPP_THROW_OUT_OF_RANGE("index out of range");

        return _data[i * _n_dim + j];
    }

    template<typename ElemType,
             storage_order SO>
    inline std::pair<size_t, size_t> 
    cpu::matrix<ElemType, SO>::
    shape() const noexcept
    {
        return {_m_dim, _n_dim};
    }

    template<typename ElemType,
             storage_order SO>
    inline ElemType const*
    cpu::matrix<ElemType, SO>::
    get_data() const
    {
        return _data;
    }

    template<typename ElemType,
             storage_order SO>
    inline ElemType*
    cpu::matrix<ElemType, SO>::
    get_data_mutable() const
    {
        return _data;
    }

    template<typename ElemType,
             storage_order SO>
    cpu::matrix<ElemType, SO>::
    ~matrix() noexcept
    {
        (void)free(_data);
    }
}
