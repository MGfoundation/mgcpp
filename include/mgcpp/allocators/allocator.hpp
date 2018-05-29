#ifndef MGCPP_ALLOCATOR_HPP
#define MGCPP_ALLOCATOR_HPP

#include <mgcpp/type_traits/device_value_type.hpp>

namespace mgcpp {

class memory_resource;
class device_memory_resource;

template <typename Type>
class allocator {
  memory_resource* m_host_resource;
  device_memory_resource* m_device_resource;

 public:
  using host_value_type = Type;
  using host_pointer = host_value_type*;
  using host_const_pointer = host_value_type const*;

  using device_value_type = typename device_value_type<Type>::type;
  using device_pointer = device_value_type*;
  using device_const_pointer = device_value_type const*;

  template <typename NewType>
  using rebind_alloc = allocator<NewType>;

  allocator();
  allocator(memory_resource* host, device_memory_resource* device);

  template <class U>
  allocator(const allocator<U>& other);

  host_pointer allocate_host(size_t n);
  void deallocate_host(host_pointer p, size_t n);

  device_pointer allocate_device(size_t n);
  void deallocate_device(device_pointer p, size_t n);

  void copy_from_host(device_pointer device,
                      device_const_pointer host,
                      size_t n) const;

  void copy_to_host(device_pointer host,
                    device_const_pointer device,
                    size_t n) const;

  // Return a default-constructed instance of this class
  allocator select_on_container_copy_construction() const;

  memory_resource* host_resource() const noexcept;
  device_memory_resource* device_resource() const noexcept;
  size_t device_id() const noexcept;
};

template <typename T1, typename T2>
bool operator==(allocator<T1> const& a, allocator<T2> const& b);

template <typename T1, typename T2>
bool operator!=(allocator<T1> const& a, allocator<T2> const& b);

}  // namespace mgcpp

#include <mgcpp/allocators/allocator.tpp>
#endif  // MGCPP_ALLOCATOR_HPP
