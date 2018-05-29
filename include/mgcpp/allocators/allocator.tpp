#include <mgcpp/allocators/allocator.hpp>
#include <mgcpp/allocators/cudamalloc_resource.hpp>
#include <mgcpp/allocators/new_delete_resource.hpp>
#include <mgcpp/allocators/device_memory_resource.hpp>

namespace mgcpp {

template <typename Type>
allocator<Type>::allocator()
    : m_host_resource(new_delete_resource::instance()),
      m_device_resource(cudamalloc_resource::instance(0)) {}

template <typename Type>
allocator<Type>::allocator(memory_resource* host,
                           device_memory_resource* device)
    : m_host_resource(host), m_device_resource(device) {}

template <typename Type>
template <class U>
allocator<Type>::allocator(const allocator<U>& other)
    : m_host_resource(other.m_host_resource),
      m_device_resource(other.m_device_resource) {}

template <typename Type>
typename allocator<Type>::host_pointer allocator<Type>::allocate_host(
    size_t n) {
  return static_cast<host_pointer>(
      m_host_resource->allocate(n * sizeof(host_value_type)));
}

template <typename Type>
void allocator<Type>::deallocate_host(allocator::host_pointer p, size_t n) {
  m_host_resource->deallocate(p, n * sizeof(host_value_type));
}

template <typename Type>
typename allocator<Type>::device_pointer allocator<Type>::allocate_device(
    size_t n) {
  return static_cast<device_pointer>(
      m_device_resource->allocate(n * sizeof(device_value_type)));
}

template <typename Type>
void allocator<Type>::deallocate_device(allocator<Type>::device_pointer p,
                                        size_t n) {
  m_device_resource->deallocate(p, n * sizeof(device_value_type));
}

template <typename Type>
void allocator<Type>::copy_from_host(allocator<Type>::device_pointer device,
                                     allocator<Type>::device_const_pointer host,
                                     size_t n) const {
  auto set_device_stat = cuda_set_device(device_id());
  if (!set_device_stat) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
  }

  auto cpy_stat =
      cuda_memcpy(device, host, n, cuda_memcpy_kind::host_to_device);

  if (!cpy_stat) {
    MGCPP_THROW_SYSTEM_ERROR(cpy_stat.error());
  }
}

template <typename Type>
void allocator<Type>::copy_to_host(allocator<Type>::device_pointer host,
                                   allocator<Type>::device_const_pointer device,
                                   size_t n) const {
  auto set_device_stat = cuda_set_device(device_id());
  if (!set_device_stat) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
  }

  auto cpy_stat =
      cuda_memcpy(host, device, n, cuda_memcpy_kind::device_to_host);

  if (!cpy_stat) {
    MGCPP_THROW_SYSTEM_ERROR(cpy_stat.error());
  }
}

template <typename Type>
allocator<Type> allocator<Type>::select_on_container_copy_construction() const {
  return allocator<Type>{};
}

template <typename Type>
device_memory_resource* allocator<Type>::device_resource() const noexcept {
  return m_device_resource;
}

template <typename Type>
size_t allocator<Type>::device_id() const noexcept {
  return device_resource()->device_id();
}

template <typename Type>
memory_resource* allocator<Type>::host_resource() const noexcept {
  return m_host_resource;
}

template <typename T1, typename T2>
bool operator==(const allocator<T1>& a, const allocator<T2>& b) {
  return a.host_resource() == b.host_resource() &&
         a.device_resource() == b.device_resource();
}

template <typename T1, typename T2>
bool operator!=(const allocator<T1>& a, const allocator<T2>& b) {
  return !(a == b);
}

}  // namespace mgcpp
