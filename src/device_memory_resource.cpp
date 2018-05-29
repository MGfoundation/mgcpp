#include <mgcpp/allocators/device_memory_resource.hpp>

namespace mgcpp {

device_memory_resource::device_memory_resource(size_t device_id)
    : _device_id(device_id) {}

size_t device_memory_resource::device_id() const {
  return _device_id;
}

}  // namespace mgcpp
