#include <mgcpp/allocators/cudamalloc_resource.hpp>
#include <mgcpp/system/assert.hpp>
#include <unordered_map>

namespace mgcpp {

cudamalloc_resource::cudamalloc_resource(size_t device_id)
    : device_memory_resource(device_id) {}

cudamalloc_resource* cudamalloc_resource::instance(size_t device_id) {
  static std::unordered_map<size_t, cudamalloc_resource> map{};
  auto it = map.find(device_id);
  if (it != map.end()) {
    return &it->second;
  } else {
    auto result =
        map.emplace(std::make_pair(device_id, cudamalloc_resource(device_id)));
    MGCPP_ASSERT(result.second,
                 "Could not emplace cuda_resource into internal map");
    return &result.first->second;
  }
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
}

bool cudamalloc_resource::do_is_equal(const memory_resource& other) const noexcept {
  const cudamalloc_resource* other_p = dynamic_cast<const cudamalloc_resource*>(&other);
  // compare if it has the same device id
  if (other_p) {
    return device_id() == other_p->device_id();
  } else {
    return false;
  }
}

}  // namespace mgcpp
