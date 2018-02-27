
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/kernels/mgblas_helpers.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/system/pun_cast.hpp>

#include <iostream>
#include <type_traits>

namespace mgcpp {
template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc>::device_matrix() noexcept
    : _context(&global_context::get_thread_context()),
      _shape{0, 0},
      _allocator(),
      _data(nullptr),
      _capacity(0) {}

template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc>::device_matrix(Alloc const& alloc)
    : _context(&global_context::get_thread_context()),
      _shape{0, 0},
      _allocator(alloc),
      _data(nullptr),
      _capacity(0) {}

template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc>::device_matrix(shape_type shape,
                                                    Alloc const& alloc)
    : _context(&global_context::get_thread_context()),
      _shape(shape),
      _allocator(alloc),
      _data(_allocator.device_allocate(_shape[0] * _shape[1])),
      _capacity(_shape[0] * _shape[1]) {}

template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc>::device_matrix(shape_type shape,
                                                    value_type init,
                                                    Alloc const& alloc)
    : _context(&global_context::get_thread_context()),
      _shape(shape),
      _allocator(alloc),
      _data(_allocator.device_allocate(_shape[0] * _shape[1])),
      _capacity(_shape[0] * _shape[1]) {
  size_t total_size = _shape[0] * _shape[1];

  auto status =
      mgblas_fill(_data, *mgcpp::pun_cast<device_pointer>(&init), total_size);
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }
}

template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc>::device_matrix(shape_type shape,
                                                    const_pointer data,
                                                    Alloc const& alloc)
    : _context(&global_context::get_thread_context()),
      _shape(shape),
      _allocator(alloc),
      _data(_allocator.device_allocate(_shape[0] * _shape[1])),
      _capacity(_shape[0] * _shape[1]) {
  size_t total_size = _shape[0] * _shape[1];

  try {
    _allocator.copy_from_host(_data, pun_cast<const_device_pointer>(data),
                              total_size);
  } catch (std::system_error const& err) {
    _allocator.device_deallocate(_data, total_size);
    MGCPP_THROW_SYSTEM_ERROR(err);
  }
}

template <typename Type, size_t DeviceId, typename Alloc>
size_t device_matrix<Type, DeviceId, Alloc>::determine_ndim(
    std::initializer_list<std::initializer_list<value_type>> const&
        list) noexcept {
  auto max_elem = std::max(list.begin(), list.end(),
                           [](auto const& first, auto const& second) {
                             return first->size() < second->size();
                           });

  if (max_elem == list.end())
    return list.begin()->size();
  else
    return max_elem->size();
}

template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc>
device_matrix<Type, DeviceId, Alloc>::from_list(
    std::initializer_list<std::initializer_list<value_type>> const& init_list,
    Alloc const& alloc) {
  auto shape = make_shape(init_list.size(), determine_ndim(init_list));

  size_t total_size = shape[0] * shape[1];

  Alloc allocator = alloc;
  pointer buffer = allocator.allocate(total_size);

  size_t i = 0;
  for (const auto& row : init_list) {
    // std::fill(std::copy(row_list.begin(),
    //                     row_list.end(),
    //                     buffer + i * _shape[1] ),
    //           buffer + (i + 1) * _shape[1],
    //           Type());
    // ++i;
    size_t j = 0;
    for (Type elem : row) {
      buffer[i + shape[0] * j] = elem;
      ++j;
    }
    ++i;
  }

  device_matrix mat(shape, buffer, alloc);

  allocator.deallocate(buffer, total_size);

  return mat;
}

template <typename Type, size_t DeviceId, typename Alloc>
template <size_t S1, size_t S2>
inline device_matrix<Type, DeviceId, Alloc>
device_matrix<Type, DeviceId, Alloc>::from_c_array(Type (&arr)[S1][S2],
                                                   Alloc const& alloc) {
  auto shape = make_shape(S1, S2);

  size_t total_size = shape[0] * shape[1];

  Alloc allocator = alloc;
  pointer buffer = allocator.allocate(total_size);

  for (size_t i = 0; i < S1; ++i) {
    for (size_t j = 0; j < S2; ++j) {
      buffer[i + shape[0] * j] = arr[i][j];
    }
  }
  device_matrix mat(shape, buffer, alloc);

  allocator.deallocate(buffer, total_size);

  return mat;
}

template <typename Type, size_t DeviceId, typename Alloc>
template <typename HostMat, typename>
device_matrix<Type, DeviceId, Alloc>::device_matrix(HostMat const& host_mat,
                                                    Alloc const& alloc)
    : _context(&global_context::get_thread_context()),
      _shape{0, 0},
      _allocator(alloc),
      _data(nullptr),
      _capacity(0) {
  adapter<HostMat> adapt{};

  pointer host_p;
  adapt(host_mat, &host_p, &_shape[0], &_shape[1]);

  size_t total_size = _shape[0] * _shape[1];
  _data = _allocator.device_allocate(total_size);
  _capacity = total_size;
  _allocator.copy_from_host(_data, pun_cast<device_pointer>(host_p),
                            total_size);
}

template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc>::device_matrix(
    device_matrix<Type, DeviceId, Alloc> const& other)
    : _context(&global_context::get_thread_context()),
      _shape(other._shape),
      _allocator(),
      _data(_allocator.device_allocate(_shape[0] * _shape[1])),
      _capacity(_shape[0] * _shape[1]) {
  auto cpy_result = cuda_memcpy(_data, other._data, _shape[0] * _shape[1],
                                cuda_memcpy_kind::device_to_device);

  if (!cpy_result) {
    _allocator.device_deallocate(_data, _capacity);
    MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
  }
}

template <typename Type, size_t DeviceId, typename Alloc>
template <typename DenseMatrix>
device_matrix<Type, DeviceId, Alloc>::device_matrix(
    dense_matrix<DenseMatrix, Type, DeviceId> const& other)
    : _context(&global_context::get_thread_context()),
      _shape((~other)._shape),
      _allocator(),
      _data(_allocator.device_allocate(_shape[0] * _shape[1])),
      _capacity(_shape[0] * _shape[1]) {
  auto cpy_result = cuda_memcpy(_data, (~other)._data, _shape[0] * _shape[1],
                                cuda_memcpy_kind::device_to_device);

  if (!cpy_result) {
    _allocator.device_deallocate(_data, _capacity);
    MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
  }
}

template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc>::device_matrix(
    device_matrix<Type, DeviceId, Alloc>&& other) noexcept
    : _context(&global_context::get_thread_context()),
      _shape(std::move(other._shape)),
      _allocator(std::move(other._allocator)),
      _data(other._data),
      _capacity(other._capacity) {
  other._data = nullptr;
  other._capacity = 0;
}

template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc>& device_matrix<Type, DeviceId, Alloc>::
operator=(device_matrix<Type, DeviceId, Alloc> const& other) {
  auto shape = other._shape;
  size_t other_size = shape[0] * shape[1];
  if (other_size > _capacity) {
    if (_data) {
      _allocator.device_deallocate(_data, _capacity);
      _capacity = 0;
    }
    _data = _allocator.device_allocate(other_size);
    _capacity = other_size;
  }
  auto cpy_result = cuda_memcpy(_data, other._data, other_size,
                                cuda_memcpy_kind::device_to_device);
  _shape = other._shape;

  if (!cpy_result) {
    MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
  }

  return *this;
}

template <typename Type, size_t DeviceId, typename Alloc>
template <typename DenseMatrix>
device_matrix<Type, DeviceId, Alloc>& device_matrix<Type, DeviceId, Alloc>::
operator=(dense_matrix<DenseMatrix, Type, DeviceId> const& other) {
  auto const& other_densemat = ~other;

  auto shape = other_densemat._shape;
  size_t other_size = shape[0] * shape[1];
  if (other_size > _capacity) {
    if (_data) {
      _allocator.device_deallocate(_data, _capacity);
      _capacity = 0;
    }
    _data = _allocator.device_allocate(other_size);
    _capacity = other_size;
  }
  auto cpy_result = cuda_memcpy(_data, other_densemat._data, other_size,
                                cuda_memcpy_kind::device_to_device);
  _shape = other_densemat._shape;

  if (!cpy_result) {
    MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
  }

  return *this;
}

template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc>& device_matrix<Type, DeviceId, Alloc>::
operator=(device_matrix<Type, DeviceId, Alloc>&& other) noexcept {
  if (_data) {
    try {
      _allocator.device_deallocate(_data, _capacity);
    } catch (...) {
    };
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

template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc>&
device_matrix<Type, DeviceId, Alloc>::resize(shape_type shape) {
  size_t total_size = shape[0] * shape[1];
  if (total_size > _capacity) {
    if (_data) {
      _allocator.device_deallocate(_data, _capacity);
      _capacity = 0;
    }
    _data = _allocator.device_allocate(total_size);
    _capacity = total_size;
  }

  _shape = shape;

  return *this;
}

template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc>&
device_matrix<Type, DeviceId, Alloc>::resize(shape_type shape,
                                             value_type init) {
  size_t total_size = shape[0] * shape[1];
  if (total_size > _capacity) {
    if (_data) {
      _allocator.device_deallocate(_data, _capacity);
      _capacity = 0;
    }
    _data = _allocator.device_allocate(total_size);
    _capacity = total_size;
  }

  _shape = shape;

  auto status =
      mgblas_fill(_data, *pun_cast<device_pointer>(&init), total_size);

  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  return *this;
}

template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc>&
device_matrix<Type, DeviceId, Alloc>::zero() {
  if (!_data) {
    MGCPP_THROW_RUNTIME_ERROR("gpu memory wasn't allocated");
  }

  auto set_device_stat = cuda_set_device(DeviceId);
  if (!set_device_stat) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
  }

  auto set_result =
      cuda_memset(_data, static_cast<Type>(0), _shape[0] * _shape[1]);
  if (!set_result) {
    MGCPP_THROW_SYSTEM_ERROR(set_result.error());
  }

  return *this;
}

template <typename Type, size_t DeviceId, typename Alloc>
column_view<device_matrix<Type, DeviceId, Alloc>, Type, DeviceId>
device_matrix<Type, DeviceId, Alloc>::column(size_t i) noexcept {
  return column_view<this_type, Type, DeviceId>(*this, i);
}

// template<typename Type,
//          size_t DeviceId,
//          typename Alloc>
// row_view<device_matrix<Type, DeviceId, Alloc>, Type, DeviceId>
// device_matrix<Type, DeviceId, Alloc>::
// row(size_t i) noexcept
// { return row_view<this_type, Type, DeviceId>(*this, i); }

template <typename Type, size_t DeviceId, typename Alloc>
typename device_matrix<Type, DeviceId, Alloc>::value_type
device_matrix<Type, DeviceId, Alloc>::check_value(size_t i, size_t j) const {
  if (i >= _shape[0] || j >= _shape[1]) {
    MGCPP_THROW_OUT_OF_RANGE("index out of range.");
  }

  auto set_device_stat = cuda_set_device(DeviceId);
  if (!set_device_stat) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
  }

  device_pointer from = (_data + (i + _shape[0] * j));
  value_type to;
  _allocator.copy_to_host(pun_cast<device_pointer>(&to), from, 1);

  return to;
}

template <typename Type, size_t DeviceId, typename Alloc>
void device_matrix<Type, DeviceId, Alloc>::set_value(size_t i,
                                                     size_t j,
                                                     value_type value) {
  if (i >= _shape[0] || j >= _shape[1]) {
    MGCPP_THROW_OUT_OF_RANGE("index out of range.");
  }

  auto set_device_stat = cuda_set_device(DeviceId);
  if (!set_device_stat) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
  }

  device_pointer to = (_data + (i + _shape[0] * j));
  _allocator.copy_from_host(to, pun_cast<const_device_pointer>(&value), 1);
}

template <typename Type, size_t DeviceId, typename Alloc>
void device_matrix<Type, DeviceId, Alloc>::copy_to_host(pointer host_p) const {
  size_t total_size = _shape[0] * _shape[1];
  if (!host_p) {
    MGCPP_THROW_RUNTIME_ERROR("provided pointer is null");
  }
  _allocator.copy_to_host(pun_cast<device_pointer>(host_p), _data, total_size);
}

template <typename Type, size_t DeviceId, typename Alloc>
typename device_matrix<Type, DeviceId, Alloc>::const_device_pointer
device_matrix<Type, DeviceId, Alloc>::data() const noexcept {
  return _data;
}

template <typename Type, size_t DeviceId, typename Alloc>
typename device_matrix<Type, DeviceId, Alloc>::device_pointer
device_matrix<Type, DeviceId, Alloc>::data_mutable() noexcept {
  return _data;
}

template <typename Type, size_t DeviceId, typename Alloc>
typename device_matrix<Type, DeviceId, Alloc>::device_pointer
device_matrix<Type, DeviceId, Alloc>::release_data() noexcept {
  device_pointer temp = _data;
  _data = nullptr;
  _capacity = 0;
  return temp;
}

template <typename Type, size_t DeviceId, typename Alloc>
size_t device_matrix<Type, DeviceId, Alloc>::capacity() const noexcept {
  return _capacity;
}

template <typename Type, size_t DeviceId, typename Alloc>
thread_context* device_matrix<Type, DeviceId, Alloc>::context() const noexcept {
  return _context;
}

template <typename Type, size_t DeviceId, typename Alloc>
typename device_matrix<Type, DeviceId, Alloc>::shape_type const&
device_matrix<Type, DeviceId, Alloc>::shape() const noexcept {
  return _shape;
}

template <typename Type, size_t DeviceId, typename Alloc>
Alloc& device_matrix<Type, DeviceId, Alloc>::allocator() noexcept {
  return _allocator;
}

template <typename Type, size_t DeviceId, typename Alloc>
device_matrix<Type, DeviceId, Alloc>::~device_matrix() noexcept {
  if (_data) {
    try {
      _allocator.device_deallocate(_data, _capacity);
    } catch (...) {
    };
  }
  global_context::reference_cnt_decr();
}
}  // namespace mgcpp
