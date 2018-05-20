#include <mgcpp/expressions/eval_context.hpp>

namespace mgcpp {

template <size_t Num,
          typename ResultType>
void eval_context::feed(placeholder_node<Num, ResultType>,
                        ResultType const& val) {
  _placeholders[Num] = static_any(val);
}

template <size_t PlaceholderID, typename ResultType>
auto eval_context::get_placeholder() const {
  return _placeholders.at(PlaceholderID).get<ResultType>();
};

}  // namespace mgcpp
