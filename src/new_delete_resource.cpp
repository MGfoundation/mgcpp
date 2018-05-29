#include <mgcpp/allocators/new_delete_resource.hpp>

namespace mgcpp {

void* new_delete_resource::do_allocate(size_t bytes) {
  return ::operator new(bytes);
}

void new_delete_resource::do_deallocate(void* p, size_t bytes) {
  (void)bytes;
  return ::operator delete(p);
}

bool new_delete_resource::do_is_equal(const memory_resource& other) const
    noexcept {
  (void)other;
    return true;
}

new_delete_resource* new_delete_resource::instance()
{
    static new_delete_resource resource{};
    return &resource;
}

}  // namespace mgcpp
