
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/matrix/column_view.hpp>
#include <mgcpp/system/assert.hpp>
#include <mgcpp/system/exception.hpp>

#include <algorithm>

namespace mgcpp {
template <typename DenseMat, typename Type>
column_view<DenseMat, Type>::column_view(
    dense_matrix<DenseMat, Type>& mat,
    size_t i) noexcept
    : _matrix(&(~mat)),
      _column_idx(i),
      _allocator(
          _matrix->allocator()) { /*static_assert(); IS DENSE MATRIX CHECK */
}

template <typename DenseMat, typename Type>
column_view<DenseMat, Type>::column_view(
    column_view<DenseMat, Type>&& other) noexcept
    : _matrix(other._matrix), _column_idx(other._column_idx) {}

template <typename DenseMat, typename Type>
column_view<DenseMat, Type>& column_view<DenseMat, Type>::
operator=(column_view<DenseMat, Type>&& other) noexcept {
  _matrix = other._matrix;
  _column_idx = other._column_idx;
  _allocator = std::move(other._allocator);
  other._matrix = nullptr;
  other._column_idx = 0;

  return *this;
}

template <typename DenseMat, typename Type>
column_view<DenseMat, Type>& column_view<DenseMat, Type>::
operator=(std::initializer_list<Type> const& init) {
  size_t size = _matrix->shape()[0];
  MGCPP_ASSERT(size == init.size(),
               "column view and assigned vector size doesn't match");

  Type* buffer = _allocator.allocate(init.size());
  std::copy(init.begin(), init.end(), buffer);

  auto status = cuda_memcpy(data_mutable(), buffer, size,
                            cuda_memcpy_kind::host_to_device);
  _allocator.deallocate(buffer, init.size());
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  return *this;
}

template <typename DenseMat, typename Type>
template <typename DenseVec>
column_view<DenseMat, Type>& column_view<DenseMat, Type>::
operator=(dense_vector<DenseVec, Type> const& vec) {
  auto const& dense_vec = ~vec;

  size_t size = _matrix->shape()[0];
  MGCPP_ASSERT(size == dense_vec.size(),
               "column view and assigned vector size doesn't match");

  auto status = cuda_memcpy(data_mutable(), dense_vec.data(), size,
                            cuda_memcpy_kind::device_to_device);
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  return *this;
}

template <typename DenseMat, typename Type>
void column_view<DenseMat, Type>::copy_to_host(Type* host_p) const {
  if (!host_p) {
    MGCPP_THROW_INVALID_ARGUMENT("provided pointer is null");
  }

  size_t size = _matrix->shape()[0];
  auto status =
      cuda_memcpy(host_p, data(), size, cuda_memcpy_kind::device_to_host);
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }
}

template <typename DenseMat, typename Type>
Type column_view<DenseMat, Type>::check_value(size_t i) const {
  if (i >= _matrix->shape()[0]) {
    MGCPP_THROW_OUT_OF_RANGE("index out of range");
  }

  Type return_value;
  auto status = cuda_memcpy(&return_value, data() + i, 1,
                            cuda_memcpy_kind::device_to_host);
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  return return_value;
}

template <typename DenseMat, typename Type>
inline Type const* column_view<DenseMat, Type>::data() const
    noexcept {
  auto shape = _matrix->shape();
  size_t idx = _column_idx * shape[0];

  Type const* ptr = _matrix->data() + idx;

  return ptr;
}

template <typename DenseMat, typename Type>
inline Type* column_view<DenseMat, Type>::data_mutable() noexcept {
  auto shape = _matrix->shape();
  size_t idx = _column_idx * shape[0];

  Type* ptr = _matrix->data_mutable() + idx;

  return ptr;
}

template <typename DenseMat, typename Type>
thread_context* column_view<DenseMat, Type>::context() const
    noexcept {
  return _matrix->context();
}

template <typename DenseMat, typename Type>
size_t column_view<DenseMat, Type>::shape() const noexcept {
  return _matrix->shape()[0];
}
}  // namespace mgcpp
