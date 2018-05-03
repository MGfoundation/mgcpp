
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef GENERIC_OP_HPP
#define GENERIC_OP_HPP

#include <memory>
#include <tuple>
#include <utility>

#include <mgcpp/expressions/eval_context.hpp>

namespace mgcpp {

template <typename TagType,
          TagType Tag,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NNonParameters,
          typename... OperandTypes>
struct generic_op
    : public ResultExprType<
          generic_op<TagType, Tag, ResultExprType, ResultType, NNonParameters, OperandTypes...>> {
  using result_type = ResultType;

  std::tuple<OperandTypes...> exprs;
  inline decltype(auto) first() const noexcept;
  inline decltype(auto) second() const noexcept;

  inline generic_op(OperandTypes const&... args) noexcept : exprs(args...) {}
  inline generic_op(OperandTypes&&... args) noexcept
      : exprs(std::move(args)...) {}

  inline void traverse(eval_context& ctx) const;
  inline result_type eval(eval_context& ctx) const;
};

enum class expression_type {
  DMAT_DMAT_ADD,
  DMAT_DMAT_MULT,
  DMAT_DVEC_MULT,
  DMAT_TRANSPOSE,
  DVEC_DVEC_ADD,
  SCALAR_DMAT_MULT
};

template <int PlaceholderID,
          template <typename> class ResultExprType,
          typename ResultType>
using placeholder_node = generic_op<int, PlaceholderID, ResultExprType, ResultType, 0>;

template <expression_type OpID,
          template <typename> class ResultExprType,
          typename ResultType,
          typename Expr>
using unary_op = generic_op<expression_type, OpID, ResultExprType, ResultType, 0, Expr>;

template <expression_type OpID,
          template <typename> class ResultExprType,
          typename ResultType,
          typename LhsExpr,
          typename RhsExpr>
using binary_op =
    generic_op<expression_type, OpID, ResultExprType, ResultType, 0, LhsExpr, RhsExpr>;

}  // namespace mgcpp

#endif  // GENERIC_OP_HPP
