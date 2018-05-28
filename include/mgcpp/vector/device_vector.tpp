
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/global_context.hpp>
#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/kernels/mgblas_helpers.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/system/pun_cast.hpp>
#include <mgcpp/type_traits/type_traits.hpp>
#include <mgcpp/vector/device_vector.hpp>

#include <algorithm>

namespace mgcpp {
template <typename Type, size_t DeviceId, typename Alloc>
device_vector<Type, DeviceId, Alloc>::device_vector() noexcept
    : _context(&global_context::get_thread_context()),
      _shape(0),
      _allocator(),
      _data(nullptr),
      _capacity(0) {}

template <typename Type, size_t DeviceId, typename Alloc>
device_vector<Type, DeviceId, Alloc>::device_vector(Alloc const& alloc) noexcept
    : _context(&global_context::get_thread_context()),
      _shape(0),
      _allocator(alloc),
      _data(nullptr),
      _capacity(0) {}

template <typename Type, size_t DeviceId, typename Alloc>
device_vector<Type, DeviceId, Alloc>::device_vector(size_t size,
                                                    Alloc const& alloc)
    : _context(&global_context::get_thread_context()),
      _shape(size),
      _allocator(alloc),
      _data(_allocator.device_allocate(_shape)),
      _capacity(_shape) {}

template <typename Type, size_t DeviceId, typename Alloc>
device_vector<Type, DeviceId, Alloc>::device_vector(size_t size,
                                                    value_type init,
                                                    Alloc const& alloc)
    : _context(&global_context::get_thread_context()),
      _shape(size),
      _allocator(alloc),
      _data(_allocator.device_allocate(_shape)),
      _capacity(_shape) {
  auto status =
      mgblas_fill(_data, *mgcpp::pun_cast<device_pointer>(&init), _shape);
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }
}

template <typename Type, size_t DeviceId, typename Alloc>
device_vector<Type, DeviceId, Alloc>::device_vector(size_t size,
                                                    const_pointer data,
                                                    Alloc const& alloc)
    : _context(&global_context::get_thread_context()),
      _shape(size),
      _allocator(alloc),
      _data(_allocator.device_allocate(_shape)),
      _capacity(size) {
  try {
    _allocator.copy_from_host(_data, pun_cast<const_device_pointer>(data),
                              _shape);
  } catch (std::system_error const& err) {
    _allocator.device_deallocate(_data, _capacity);
    MGCPP_THROW_SYSTEM_ERROR(err);
  }
}

template <typename Type, size_t DeviceId, typename Alloc>
device_vector<Type, DeviceId, Alloc>::device_vector(
    std::initializer_list<value_type> const& array,
    Alloc const& alloc)
    : _context(&global_context::get_thread_context()),
      _shape(array.size()),
      _allocator(alloc),
      _data(_allocator.device_allocate(_shape)),
      _capacity(_shape) {
  try {
    // std::initializer_list's members are guaranteed to be
    // contiguous in memory: from C++11 § [support.initlist] 18.9/1
    _allocator.copy_from_host(
        _data, pun_cast<const_device_pointer>(array.begin()), _shape);
  } catch (std::system_error const& err) {
    _allocator.device_deallocate(_data, _capacity);
    MGCPP_THROW_SYSTEM_ERROR(err);
  }
}

template <typename Type, size_t DeviceId, typename Alloc>
template <typename HostVec, typename>
device_vector<Type, DeviceId, Alloc>::device_vector(HostVec const& host_mat,
                                                    Alloc const& alloc)
    : _context(&global_context::get_thread_context()),
      _shape(0),
      _allocator(alloc),
      _data(nullptr),
      _capacity(0) {
  adapter<HostVec> adapt{};

  pointer host_p;
  adapt(host_mat, &host_p, &_shape);
  _capacity = _shape;
  _data = _allocator.device_allocate(_shape);

  try {
    _allocator.copy_from_host(_data, pun_cast<const_device_pointer>(host_p),
                              _shape);
  } catch (std::system_error const& err) {
    _allocator.device_deallocate(_data, _capacity);
    MGCPP_THROW_SYSTEM_ERROR(err);
  }
}

template <typename Type, size_t DeviceId, typename Alloc>
device_vector<Type, DeviceId, Alloc>::device_vector(
    device_vector<Type, DeviceId, Alloc> const& other)
    : _context(&global_context::get_thread_context()),
      _shape(other._shape),
      _allocator(),
      _data(_allocator.device_allocate(_shape)),
      _capacity(_shape) {
  auto cpy_result = cuda_memcpy(_data, other._data, _shape,
                                cuda_memcpy_kind::device_to_device);

  if (!cpy_result) {
    _allocator.device_deallocate(_data, _shape);
    MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
  }
}

template <typename Type, size_t DeviceId, typename Alloc>
template <typename DenseVec>
device_vector<Type, DeviceId, Alloc>::device_vector(
    dense_vector<DenseVec, Type, DeviceId> const& other)
    : _context(&global_context::get_thread_context()),
      _shape((~other)._shape),
      _allocator(),
      _data(_allocator.device_allocate(_shape)),
      _capacity(_shape) {
  auto cpy_result = cuda_memcpy(_data, (~other)._data, _shape,
                                cuda_memcpy_kind::device_to_device);

  if (!cpy_result) {
    _allocator.device_deallocate(_data, _shape);
    MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
  }
}

template <typename Type, size_t DeviceId, typename Alloc>
device_vector<Type, DeviceId, Alloc>::device_vector(
    device_vector<Type, DeviceId, Alloc>&& other) noexcept
    : _context(&global_context::get_thread_context()),
      _shape(std::move(other._shape)),
      _allocator(std::move(other._allocator)),
      _data(other._data),
      _capacity(other._capacity) {
  other._data = nullptr;
  other._capacity = 0;
}

template <typename Type, size_t DeviceId, typename Alloc>
template <size_t S>
inline device_vector<Type, DeviceId, Alloc>
device_vector<Type, DeviceId, Alloc>::from_c_array(Type (&arr)[S],
                                                   Alloc const& alloc) {
  return device_vector<Type, DeviceId, Alloc>(S, arr, alloc);
}

template <typename Type, size_t DeviceId, typename Alloc>
template <typename DenseVec>
device_vector<Type, DeviceId, Alloc>& device_vector<Type, DeviceId, Alloc>::
operator=(dense_vector<DenseVec, Type, DeviceId> const& other) {
  auto const& other_densevec = ~other;

  if (other_densevec._shape > _capacity) {
    if (_data) {
      _allocator.device_deallocate(_data, _capacity);
      _capacity = 0;
    }
    _data = _allocator.device_allocate(other_densevec._shape);
    _capacity = other_densevec._shape;
  }

  auto cpy_result =
      cuda_memcpy(_data, other_densevec._data, other_densevec._shape,
                  cuda_memcpy_kind::device_to_device);
  _shape = other_densevec._shape;

  if (!cpy_result) {
    MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
  }

  return *this;
}

template <typename Type, size_t DeviceId, typename Alloc>
device_vector<Type, DeviceId, Alloc>& device_vector<Type, DeviceId, Alloc>::
operator=(device_vector<Type, DeviceId, Alloc> const& other) {
  if (other._shape > _capacity) {
    if (_data) {
      _allocator.device_deallocate(_data, _capacity);
      _capacity = 0;
    }
    _data = _allocator.device_allocate(other._shape);
    _capacity = other._shape;
  }

  auto cpy_result = cuda_memcpy(_data, other._data, other._shape,
                                cuda_memcpy_kind::device_to_device);
  _shape = other._shape;

  if (!cpy_result) {
    MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
  }

  return *this;
}

template <typename Type, size_t DeviceId, typename Alloc>
device_vector<Type, DeviceId, Alloc>& device_vector<Type, DeviceId, Alloc>::
operator=(device_vector<Type, DeviceId, Alloc>&& other) noexcept {
  if (_data) {
    try {
      _allocator.device_deallocate(_data, _shape);
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
device_vector<Type, DeviceId, Alloc>&
device_vector<Type, DeviceId, Alloc>::resize(size_t size,
                                             value_type pad_value) {
  if (size > _capacity) {
    auto new_data = _allocator.device_allocate(size);
    if (_data) {
      auto cpy_result = cuda_memcpy(new_data, _data, _shape,
                                    cuda_memcpy_kind::device_to_device);
      if (!cpy_result) {
        MGCPP_THROW_SYSTEM_ERROR(cpy_result.error());
      }
      _allocator.device_deallocate(_data, _capacity);
      _capacity = 0;
    }
    _data = new_data;
    _capacity = size;
  }
  if (size > _shape) {
    auto fill_result = mgblas_fill(_data + _shape,
                                   *mgcpp::pun_cast<device_pointer>(&pad_value),
                                   size - _shape);
    if (!fill_result) {
      MGCPP_THROW_SYSTEM_ERROR(fill_result.error());
    }
  }
  _shape = size;

  return *this;
}

template <typename Type, size_t DeviceId, typename Alloc>
device_vector<Type, DeviceId, Alloc>&
device_vector<Type, DeviceId, Alloc>::zero() {
  if (!_data) {
    MGCPP_THROW_RUNTIME_ERROR("gpu memory wasn't allocated");
  }

  auto set_device_stat = cuda_set_device(DeviceId);
  if (!set_device_stat) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
  }

  auto set_result = cuda_memset(_data, static_cast<Type>(0), _shape);
  if (!set_result) {
    MGCPP_THROW_SYSTEM_ERROR(set_result.error());
  }

  return *this;
}

template <typename Type, size_t DeviceId, typename Alloc>
void device_vector<Type, DeviceId, Alloc>::copy_to_host(pointer host_p) const {
  if (!host_p) {
    MGCPP_THROW_RUNTIME_ERROR("provided pointer is null");
  }
  _allocator.copy_to_host(pun_cast<device_pointer>(host_p), _data, _shape);
}

template <typename Type, size_t DeviceId, typename Alloc>
typename device_vector<Type, DeviceId, Alloc>::value_type
device_vector<Type, DeviceId, Alloc>::check_value(size_t i) const {
  if (i >= _shape) {
    MGCPP_THROW_OUT_OF_RANGE("index out of range.");
  }

  auto set_device_stat = cuda_set_device(DeviceId);
  if (!set_device_stat) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
  }

  device_pointer from = (_data + i);
  value_type to;
  _allocator.copy_to_host(pun_cast<device_pointer>(&to), from, 1);

  return to;
}

template <typename Type, size_t DeviceId, typename Alloc>
void device_vector<Type, DeviceId, Alloc>::set_value(size_t i,
                                                     value_type value) {
  if (i >= _shape) {
    MGCPP_THROW_OUT_OF_RANGE("index out of range.");
  }

  auto set_device_stat = cuda_set_device(DeviceId);
  if (!set_device_stat) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
  }

  device_pointer to = (_data + i);
  value_type from = value;
  _allocator.copy_from_host(to, pun_cast<device_pointer>(&from), 1);
}

template <typename Type, size_t DeviceId, typename Alloc>
typename device_vector<Type, DeviceId, Alloc>::device_pointer
device_vector<Type, DeviceId, Alloc>::release_data() noexcept {
  device_pointer temp = _data;
  _data = nullptr;
  _capacity = 0;
  return temp;
}

template <typename Type, size_t DeviceId, typename Alloc>
thread_context* device_vector<Type, DeviceId, Alloc>::context() const noexcept {
  return _context;
}

template <typename Type, size_t DeviceId, typename Alloc>
size_t device_vector<Type, DeviceId, Alloc>::capacity() const noexcept {
  return _capacity;
}

template <typename Type, size_t DeviceId, typename Alloc>
typename device_vector<Type, DeviceId, Alloc>::const_device_pointer
device_vector<Type, DeviceId, Alloc>::data() const noexcept {
  return _data;
}

template <typename Type, size_t DeviceId, typename Alloc>
typename device_vector<Type, DeviceId, Alloc>::device_pointer
device_vector<Type, DeviceId, Alloc>::data_mutable() noexcept {
  return _data;
}

template <typename Type, size_t DeviceId, typename Alloc>
size_t device_vector<Type, DeviceId, Alloc>::size() const noexcept {
  return _shape;
}

template <typename Type, size_t DeviceId, typename Alloc>
typename device_vector<Type, DeviceId, Alloc>::shape_type
device_vector<Type, DeviceId, Alloc>::shape() const noexcept {
  return mgcpp::make_shape(_shape);
}

template <typename Type, size_t DeviceId, typename Alloc>
Alloc& device_vector<Type, DeviceId, Alloc>::allocator() noexcept {
  return _allocator;
}

template <typename Type, size_t DeviceId, typename Alloc>
device_vector<Type, DeviceId, Alloc>::~device_vector() noexcept {
  if (_data) {
    try {
      _allocator.device_deallocate(_data, _capacity);
    } catch (...) {
    };
  }
  global_context::reference_cnt_decr();
}
}  // namespace mgcpp
