
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
             storage_order StoreOrder>
    cpu::matrix<ElemType, StoreOrder>::
    matrix()
        : _data(nullptr),
          _row_dim(0),
          _column_dim(0) {}

    template<typename ElemType,
             storage_order StoreOrder>
    cpu::matrix<ElemType, StoreOrder>::
    matrix(size_t i, size_t j)
        : _data(nullptr),
          _row_dim(j),
          _column_dim(i)
    {
        size_t total_size = i * j;
        ElemType* ptr =
            (ElemType*)malloc(sizeof(ElemType) * total_size);

        if(!ptr)
            MGCPP_THROW_BAD_ALLOC;
        
        _data = ptr;
    }

    template<typename ElemType,
             storage_order StoreOrder>
    cpu::matrix<ElemType, StoreOrder>::
    matrix(size_t i, size_t j, ElemType init)
        : _data(nullptr),
          _row_dim(j),
          _column_dim(i)
    {
        size_t total_size = i * j;
        ElemType* ptr =
            (ElemType*)malloc(sizeof(ElemType) * total_size);

        if(!ptr)
            MGCPP_THROW_BAD_ALLOC;

        std::fill(ptr, ptr + total_size, init);
        
        _data = ptr;
    }

    inline ElemType
    operator()(size_t i, size_t j) const
    {
        if(i > _col_dim || j > _row_dim)
                MGCPP_THROW_OUT_OF_RANGE("index out of range");

        return _data[i * row_dim + j];
    }

    inline ElemType&
    operator()(size_t i, size_t j)
    {
        if(i > _col_dim || j > _row_dim)
            MGCPP_THROW_OUT_OF_RANGE("index out of range");

        return _data[i * row_dim + j];
    }

    template<typename ElemType,
             storage_order StoreOrder>
    inline size_t
    cpu::matrix<ElemType, StoreOrder>::
    rows() const noexcept
    {
        return _row_dim;
    }

    template<typename ElemType,
             storage_order StoreOrder>
    inline size_t
    cpu::matrix<ElemType, StoreOrder>::
    columns() const noexcept
    {
        return _col_dim;
    }

    inline ElemType const*
    get_data() const
    {
        return _data;
    }

    inline ElemType*
    get_data_mutable() const
    {
        return _data;
    }
}
