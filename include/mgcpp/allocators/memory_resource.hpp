#ifndef MEMORY_RESOURCE_HPP
#define MEMORY_RESOURCE_HPP

#include <atomic>
#include <cstddef>

namespace mgcpp {

enum class byte : unsigned char {};

// Taken & modified from https://github.com/phalpern/CppCon2017Code
class memory_resource {
 public:
  virtual ~memory_resource() = default;

  void* allocate(size_t bytes) { return do_allocate(bytes); }
  void deallocate(void* p, size_t bytes) { return do_deallocate(p, bytes); }

  // `is_equal` is needed because polymorphic allocators are sometimes
  // produced as a result of type erasure.  In that case, two different
  // instances of a polymorphic_memory_resource may actually represent
  // the same underlying allocator and should compare equal, even though
  // their addresses are different.
  bool is_equal(const memory_resource& other) const noexcept {
    return do_is_equal(other);
  }

 protected:
  virtual void* do_allocate(size_t bytes) = 0;
  virtual void do_deallocate(void* p, size_t bytes) = 0;
  virtual bool do_is_equal(const memory_resource& other) const noexcept = 0;
};

inline bool operator==(const memory_resource& a, const memory_resource& b) {
  // Call `is_equal` rather than using address comparisons because some
  // polymorphic allocators are produced as a result of type erasure.  In
  // that case, `a` and `b` may contain `memory_resource`s with different
  // addresses which, nevertheless, should compare equal.
  return &a == &b || a.is_equal(b);
}

inline bool operator!=(const memory_resource& a, const memory_resource& b) {
  return !(a == b);
}

}  // namespace mgcpp

#endif  // MEMORY_RESOURCE_HPP
