
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/matrix/column_view.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/system/assert.hpp>

#include <algorithm>

namespace mgcpp
{
    template<typename DenseMat,
             typename Type,
             size_t DeviceId>
    column_view<DenseMat, Type, DeviceId>::
    column_view(dense_matrix<DenseMat, Type, DeviceId>& mat, size_t i) noexcept
        : _matrix(&(~mat)),
          _column_idx(i),
          _allocator(_matrix->allocator())
    { /*static_assert(); IS DENSE MATRIX CHECK */ }

    template<typename DenseMat,
             typename Type,
             size_t DeviceId>
    column_view<DenseMat, Type, DeviceId>::
    column_view(column_view<DenseMat, Type, DeviceId>&& other) noexcept
        : _matrix(other._matrix),
          _column_idx(other._column_idx) {}

    template<typename DenseMat,
             typename Type,
             size_t DeviceId>
    column_view<DenseMat, Type, DeviceId>&
    column_view<DenseMat, Type, DeviceId>::
    operator=(column_view<DenseMat, Type, DeviceId>&& other) noexcept
    {
        _matrix = other._matrix;
        _column_idx = other._column_idx;
        _allocator = std::move(other._allocator);
        other._matrix = nullptr;
        other._column_idx = 0;

        return *this;
    }

    template<typename DenseMat,
             typename Type,
             size_t DeviceId>
    column_view<DenseMat, Type, DeviceId>&
    column_view<DenseMat, Type, DeviceId>::
    operator=(std::initializer_list<Type> const& init)
    {
        size_t size = _matrix->shape().first;
        MGCPP_ASSERT(size == init.size(),
                     "column view and assigned vector size doesn't match");

        Type* buffer = _allocator.allocate(init.size());
        std::copy(init.begin(), init.end(), buffer);

        auto status = cuda_memcpy(data_mutable(),
                                  buffer,
                                  size,
                                  cuda_memcpy_kind::host_to_device);
        free(buffer);
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return *this;
    }

    template<typename DenseMat,
             typename Type,
             size_t DeviceId>
    template<typename DenseVec>
    column_view<DenseMat, Type, DeviceId>&
    column_view<DenseMat, Type, DeviceId>::
    operator=(dense_vector<DenseVec, Type, column, DeviceId> const& vec)
    {
        auto const& dense_vec = ~vec;

        size_t size = _matrix->shape().first;
        MGCPP_ASSERT(size == dense_vec.shape(),
                     "column view and assigned vector size doesn't match");

        auto status = cuda_memcpy(data_mutable(),
                                  dense_vec.data(),
                                  size,
                                  cuda_memcpy_kind::device_to_device);
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return *this;
    }

    template<typename DenseMat,
             typename Type,
             size_t DeviceId>
    void
    column_view<DenseMat, Type, DeviceId>::
    copy_to_host(Type* host_p) const
    {
        if(!host_p)
        { MGCPP_THROW_INVALID_ARGUMENT("provided pointer is null"); }

        size_t size = _matrix->shape().first;
        auto status = cuda_memcpy(host_p,
                                  data(),
                                  size,
                                  cuda_memcpy_kind::device_to_host);
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }
    }

    template<typename DenseMat,
             typename Type,
             size_t DeviceId>
    Type
    column_view<DenseMat, Type, DeviceId>::
    check_value(size_t i) const
    {
        if(i >= _matrix->shape().first)
        { MGCPP_THROW_OUT_OF_RANGE("index out of range"); }

        Type return_value;
        auto status = cuda_memcpy(&return_value,
                                  data() + i,
                                  1,
                                  cuda_memcpy_kind::device_to_host);
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return return_value;
    }

    template<typename DenseMat,
             typename Type,
             size_t DeviceId>
    inline Type const*
    column_view<DenseMat, Type, DeviceId>::
    data() const noexcept
    {
        auto shape = _matrix->shape();
        size_t idx = _column_idx * shape.first;

        Type const* ptr = _matrix->data() + idx;

        return ptr;
    }

    template<typename DenseMat,
             typename Type,
             size_t DeviceId>
    inline Type*
    column_view<DenseMat, Type, DeviceId>::
    data_mutable() noexcept
    {
        auto shape = _matrix->shape();
        size_t idx = _column_idx * shape.first;

        Type* ptr = _matrix->data_mutable() + idx;

        return ptr;
    }

    template<typename DenseMat,
             typename Type,
             size_t DeviceId>
    thread_context*
    column_view<DenseMat, Type, DeviceId>::
    context() const noexcept
    { return _matrix->context(); }

    template<typename DenseMat,
             typename Type,
             size_t DeviceId>
    size_t 
    column_view<DenseMat, Type, DeviceId>::
    shape() const noexcept
    { return _matrix->shape().second; }
}
