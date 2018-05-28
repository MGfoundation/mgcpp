
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef GENERIC_OP_HPP
#define GENERIC_OP_HPP

#include <memory>
#include <tuple>
#include <utility>

#include <mgcpp/expressions/forward.hpp>
#include <mgcpp/expressions/placeholder.hpp>
#include <mgcpp/global/shape.hpp>
#include <mgcpp/global/tuple_utils.hpp>
#include <mgcpp/type_traits/shape_type.hpp>

namespace mgcpp {

template <typename TagType,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
struct generic_expr : public ResultExprType<TagType> {
  // Type of self
  using this_type = generic_expr<TagType,
                                 ResultExprType,
                                 ResultType,
                                 NParameters,
                                 OperandTypes...>;
  using tag_type = TagType;

  // The resulting type from eval()-ing this node (i.e. device_matrix<float>)
  using result_type = ResultType;

  template <typename T>
  using result_expr_type = ResultExprType<T>;

  // Is this node a terminal node (i.e. with no child nodes)
  static constexpr bool is_terminal = sizeof...(OperandTypes) == NParameters;
  static constexpr size_t n_parameters = NParameters;

  // Operand expressions (first NParameter elements are non-expression
  // parameters)
  std::tuple<OperandTypes...> exprs;

  // Returns tuple containing operands
  inline decltype(auto) operands() const noexcept;

  // Returns tuple containing parameters
  inline decltype(auto) parameters() const noexcept;

  // Convenience getters for the operands
  inline decltype(auto) first() const noexcept;
  inline decltype(auto) second() const noexcept;

  // Constructor
  inline generic_expr(OperandTypes... args) noexcept;

  // Build cache info by traversing the entire graph
  inline void traverse() const;

  /**
   * Evaluate this expression with an empty default context.
   */
  inline ResultType eval() const;

  /**
   * Evaluate this expression with context.
   * \param ctx the context the expression is evaluated in.
   */
  inline ResultType eval(eval_context const& ctx) const;

  /**
   * Get the shape of this expression without computing the whole expression.
   */
  template <
      typename T = ResultType,
      typename = typename std::enable_if<std::is_same<T, ResultType>::value &&
                                         mgcpp::has_shape<T>::value>::type>
  inline typename T::shape_type shape(eval_context const& ctx) const;
};

// A unary operator with 1 operand (i.e. map)
template <typename TagType,
          template <typename> class ResultExprType,
          typename ResultType,
          typename Expr>
using unary_expr =
    generic_expr<TagType, ResultExprType, ResultType, 0, Expr>;

// A binary operator with left and right operands (i.e. addition,
// multiplication)
template <typename TagType,
          template <typename> class ResultExprType,
          typename ResultType,
          typename LhsExpr,
          typename RhsExpr>
using binary_expr =
    generic_expr<TagType, ResultExprType, ResultType, 0, LhsExpr, RhsExpr>;
}  // namespace mgcpp

#include <mgcpp/expressions/generic_expr.tpp>
#endif  // GENERIC_OP_HPP
