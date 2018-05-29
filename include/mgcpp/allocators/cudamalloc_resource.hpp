#ifndef CUDA_RESOURCE_HPP
#define CUDA_RESOURCE_HPP

#include <mgcpp/allocators/device_memory_resource.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cuda/memory.hpp>

namespace mgcpp {

class cudamalloc_resource final : public device_memory_resource {

public:
  static cudamalloc_resource* instance(size_t device_id);

 protected:
  void* do_allocate(size_t bytes) override;

  void do_deallocate(void* p, size_t bytes) override;

  bool do_is_equal(const memory_resource& other) const noexcept override;

 private:
  explicit cudamalloc_resource(size_t device_id);
};

}  // namespace mgcpp

#endif  // CUDA_RESOURCE_HPP
