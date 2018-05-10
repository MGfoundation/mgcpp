#ifndef SHAPE_EVALUATOR_HPP
#define SHAPE_EVALUATOR_HPP

#include <mgcpp/expressions/forward.hpp>

namespace mgcpp {
struct shape_evaluator {
  template <typename Op>
  inline static typename Op::result_type::shape_type shape(Op const& op, eval_context const& ctx);
};
}  // namespace mgcpp

#include <mgcpp/expressions/shape_evaluator.tpp>

#endif // SHAPE_EVALUATOR_HPP
