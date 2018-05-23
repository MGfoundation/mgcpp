#ifndef SHAPE_EVALUATOR_HPP
#define SHAPE_EVALUATOR_HPP

#include <mgcpp/expressions/forward.hpp>

namespace mgcpp {
namespace shape_evaluator {
template <typename T>
inline typename T::result_type::shape_type shape(expression<T> const& op,
                                                  eval_context const& ctx);
};
}  // namespace mgcpp

#include <mgcpp/expressions/shape_evaluator.tpp>

#endif  // SHAPE_EVALUATOR_HPP
