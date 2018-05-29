#include <mgcpp/allocators/cudamalloc_resource.hpp>
#include <mgcpp/system/assert.hpp>
#include <vector>

namespace mgcpp {

cudamalloc_resource::cudamalloc_resource(size_t device_id)
    : device_memory_resource(device_id) {}

cudamalloc_resource* cudamalloc_resource::instance(size_t device_id) {
  static std::vector<std::unique_ptr<cudamalloc_resource>> resources([] {
    int device_number = 0;
    std::error_code status = cudaGetDeviceCount(&device_number);
    if (status != status_t::success) {
      MGCPP_THROW_SYSTEM_ERROR(status);
    }
    std::vector<std::unique_ptr<cudamalloc_resource>> vec;
    for (size_t i = 0; i < static_cast<size_t>(device_number); ++i) {
        vec.emplace_back(new cudamalloc_resource(i));
    }
    return vec;
  }());

  if (device_id >= resources.size()) {
    MGCPP_THROW_OUT_OF_RANGE("Invalid device id.");
  }
  return resources[device_id].get();
}

void* cudamalloc_resource::do_allocate(size_t bytes) {
  auto set_device_stat = cuda_set_device(device_id());
  if (!set_device_stat) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
  }

  auto ptr = cuda_malloc<byte>(bytes);
  if (!ptr) {
    MGCPP_THROW_SYSTEM_ERROR(ptr.error());
  }
  _allocated_bytes += bytes;
  return ptr.value();
}

void cudamalloc_resource::do_deallocate(void* p, size_t bytes) {
  (void)bytes;
  auto set_device_stat = cuda_set_device(device_id());
  if (!set_device_stat) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_stat.error());
  }

  auto free_stat = cuda_free(p);
  if (!p) {
    MGCPP_THROW_SYSTEM_ERROR(free_stat.error());
  }

  _allocated_bytes -= bytes;
}

bool cudamalloc_resource::do_is_equal(const memory_resource& other) const
    noexcept {
  const cudamalloc_resource* other_p =
      dynamic_cast<const cudamalloc_resource*>(&other);
  // compare if it has the same device id
  if (other_p) {
    return device_id() == other_p->device_id();
  } else {
    return false;
  }
}

size_t cudamalloc_resource::allocated_bytes() const noexcept {
  return _allocated_bytes;
}

}  // namespace mgcpp
