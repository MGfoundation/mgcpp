#include <mgcpp/expressions/eval_context.hpp>

namespace mgcpp {

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
