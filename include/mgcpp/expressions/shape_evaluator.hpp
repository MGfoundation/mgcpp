#ifndef SHAPE_EVALUATOR_HPP
#define SHAPE_EVALUATOR_HPP

#include <mgcpp/expressions/forward.hpp>

namespace mgcpp {
namespace shape_evaluator {
template <typename Op>
inline typename Op::result_type::shape_type shape(Op const& op,
                                                  eval_context const& ctx);
};
}  // namespace mgcpp

#include <mgcpp/expressions/shape_evaluator.tpp>

#endif  // SHAPE_EVALUATOR_HPP
