#ifndef DEVICE_MEMORY_RESOURCE_HPP
#define DEVICE_MEMORY_RESOURCE_HPP

#include <mgcpp/allocators/memory_resource.hpp>

namespace mgcpp {

class device_memory_resource : public memory_resource {
  size_t _device_id;

 public:
  device_memory_resource(size_t device_id);

  size_t device_id() const;
};

}  // namespace mgcpp

#endif  // DEVICE_MEMORY_RESOURCE_HPP
