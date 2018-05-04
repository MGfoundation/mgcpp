#include <mgcpp/expressions/eval_context.hpp>

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

template <int Num,
          template <typename> class ResultExprType,
          typename ResultType>
void eval_context::feed(placeholder_node<Num, ResultExprType, ResultType>,
                        ResultType const& val) {
  _placeholders[Num] = type_erased(val);
}

template <size_t PlaceholderID, typename ResultType>
auto eval_context::get_placeholder() const {
  return _placeholders.at(PlaceholderID).get<ResultType>();
};

}  // namespace mgcpp
