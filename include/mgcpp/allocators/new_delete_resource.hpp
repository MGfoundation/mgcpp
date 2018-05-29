#ifndef NEW_DELETE_RESOURCE_HPP
#define NEW_DELETE_RESOURCE_HPP

#include <mgcpp/allocators/memory_resource.hpp>

namespace mgcpp {

class new_delete_resource final : public memory_resource {
 public:
  static new_delete_resource* instance();

 protected:
  void* do_allocate(size_t bytes) override;

  void do_deallocate(void* p, size_t bytes) override;

  bool do_is_equal(const memory_resource&) const noexcept override;

 private:
  new_delete_resource() = default;
};

}  // namespace mgcpp

#endif  // NEW_DELETE_RESOURCE_HPP
