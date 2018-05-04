#ifndef TYPE_ERASED_HPP
#define TYPE_ERASED_HPP

#include <memory>

namespace mgcpp {

struct type_erased {
  type_erased() = default;

  template <typename T>
  type_erased(T&& data);

  template <typename T>
  T get() const;

  struct concept {
    virtual ~concept() = default;
  };

  template <typename T>
  struct model final : concept {
    model(T const& x);
    model(T&& x);
    T data;
  };

  std::shared_ptr<concept const> m;
};

}

#include <mgcpp/global/type_erased.tpp>
#endif // TYPE_ERASED_HPP
