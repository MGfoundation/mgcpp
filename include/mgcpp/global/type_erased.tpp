#include <mgcpp/global/type_erased.hpp>

namespace mgcpp {
template <typename T>
type_erased::type_erased(T&& data)
    : m(std::make_shared<model<std::remove_reference_t<T>>>(
          std::forward<T>(data))) {}

template <typename T>
T type_erased::get() const {
  return static_cast<model<T> const&>(*m).data;
}

template <typename T>
type_erased::model<T>::model(T const& x) : data(x) {}

template <typename T>
type_erased::model<T>::model(T&& x) : data(std::move(x)) {}

}  // namespace mgcpp
