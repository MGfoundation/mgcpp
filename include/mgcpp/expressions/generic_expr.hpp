
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
#include <mgcpp/global/shape.hpp>
#include <mgcpp/global/tuple_utils.hpp>
#include <mgcpp/type_traits/shape_type.hpp>

namespace mgcpp {

enum class expression_type {
  DMAT_DMAT_ADD,
  DMAT_DMAT_MULT,
  DMAT_DVEC_MULT,
  DMAT_TRANSPOSE,
  DVEC_DVEC_ADD,
  SCALAR_DMAT_MULT,
  DMAT_REF,
  DVEC_REF,
  TIE,
  ALL_ZEROS,
  ALL_ONES,
  SHAPE,
  SCALAR_CONSTANT
};

template <typename TagType,
          TagType Tag,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
struct generic_expr : public ResultExprType<generic_expr<TagType,
                                                         Tag,
                                                         ResultExprType,
                                                         ResultType,
                                                         NParameters,
                                                         OperandTypes...>> {
  // Type of self
  using this_type = generic_expr<TagType,
                                 Tag,
                                 ResultExprType,
                                 ResultType,
                                 NParameters,
                                 OperandTypes...>;

  // The resulting type from eval()-ing this node (i.e. device_matrix<float>)
  using result_type = ResultType;

  // Is this node a terminal node (i.e. with no child nodes)
  static constexpr bool is_terminal = sizeof...(OperandTypes) == NParameters;

  static constexpr TagType tag = Tag;

  // Operand expressions (first NParameter elements are non-expression
  // parameters)
  std::tuple<OperandTypes...> exprs;

  // Convenience getters for the operands
  inline decltype(auto) first() const noexcept;
  inline decltype(auto) second() const noexcept;

  // Constructor
  inline generic_expr(OperandTypes... args) noexcept;

  // Analyze information about the expression tree
  inline void traverse() const;

  /**
   * Evaluate this expression with an empty default context.
   */
  inline result_type eval() const;

  /**
   * Evaluate this expression with context.
   * \param ctx the context the expression is evaluated in.
   */
  inline result_type eval(eval_context const& ctx) const;
};

// A unary operator with 1 operand (i.e. map)
template <expression_type OpID,
          template <typename> class ResultExprType,
          typename ResultType,
          typename Expr>
using unary_expr =
    generic_expr<expression_type, OpID, ResultExprType, ResultType, 0, Expr>;

// A binary operator with left and right operands (i.e. addition,
// multiplication)
template <expression_type OpID,
          template <typename> class ResultExprType,
          typename ResultType,
          typename LhsExpr,
          typename RhsExpr>
using binary_expr = generic_expr<expression_type,
                                 OpID,
                                 ResultExprType,
                                 ResultType,
                                 0,
                                 LhsExpr,
                                 RhsExpr>;
}  // namespace mgcpp

#include <mgcpp/expressions/generic_expr.tpp>
#endif  // GENERIC_OP_HPP
