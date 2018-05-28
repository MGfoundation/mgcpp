
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/allocators/default.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp {
template <typename T>
default_allocator<T>::default_allocator(size_t device_id)
    : _device_id(device_id) {}

template <typename T>
typename default_allocator<T>::pointer default_allocator<T>::allocate(
    size_t n) {
  return _alloc_tr::allocate(_alloc, n);
}

template <typename T>
void default_allocator<T>::deallocate(pointer p, size_t n) {
  return _alloc_tr::deallocate(_alloc, p, n);
}

template <typename T>
typename default_allocator<T>::device_pointer
default_allocator<T>::device_allocate(size_t n) const {
  auto set_device_stat = cuda_set_device(_device_id);
  if (!set_device_stat) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
  }

  auto ptr = cuda_malloc<device_value_type>(n);
  if (!ptr) {
    MGCPP_THROW_SYSTEM_ERROR(ptr.error());
  }
  return ptr.value();
}

template <typename T>
void default_allocator<T>::device_deallocate(device_pointer p, size_t n) const {
  (void)n;
  auto set_device_stat = cuda_set_device(_device_id);
  if (!set_device_stat) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
  }

  auto free_stat = cuda_free<device_value_type>(p);
  if (!p) {
    MGCPP_THROW_SYSTEM_ERROR(free_stat.error());
  }
}

template <typename T>
void default_allocator<T>::copy_from_host(device_pointer device,
                                          const_device_pointer host,
                                          size_t n) const {
  auto set_device_stat = cuda_set_device(_device_id);
  if (!set_device_stat) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
  }

  auto cpy_stat =
      cuda_memcpy(device, host, n, cuda_memcpy_kind::host_to_device);

  if (!cpy_stat) {
    MGCPP_THROW_SYSTEM_ERROR(cpy_stat.error());
  }
}

template <typename T>
void default_allocator<T>::copy_to_host(device_pointer host,
                                        const_device_pointer device,
                                        size_t n) const {
  auto set_device_stat = cuda_set_device(_device_id);
  if (!set_device_stat) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
  }

  auto cpy_stat =
      cuda_memcpy(host, device, n, cuda_memcpy_kind::device_to_host);

  if (!cpy_stat) {
    MGCPP_THROW_SYSTEM_ERROR(cpy_stat.error());
  }
}

template <typename T>
size_t default_allocator<T>::device_id() const noexcept {
  return _device_id;
}
}  // namespace mgcpp
