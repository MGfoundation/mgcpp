#include <mgcpp/global/type_erased.hpp>

namespace mgcpp {
template <typename T>
static_any::static_any(T data)
    : m(std::make_shared<model<T>>(
          std::move(data))) {}

template <typename T>
T static_any::get() const {
  return static_cast<model<T> const&>(*m).data;
}

template <typename T>
static_any::model<T>::model(T x) : data(std::move(x)) {}

}  // namespace mgcpp
