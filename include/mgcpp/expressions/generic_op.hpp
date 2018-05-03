
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef GENERIC_OP_HPP
#define GENERIC_OP_HPP

#include <memory>
#include <utility>
#include <tuple>

#include <mgcpp/expressions/eval_context.hpp>

namespace mgcpp {

template <int OpID,
          template <typename> class ResultExprType,
          typename ResultType,
          typename... OperandTypes>
struct generic_op
    : public ResultExprType<
          generic_op<OpID, ResultExprType, ResultType, OperandTypes...>> {
  using result_type = ResultType;

  std::tuple<OperandTypes...> exprs;
  inline decltype(auto) first() const noexcept;
  inline decltype(auto) second() const noexcept;

  inline generic_op(OperandTypes const& ... args) noexcept
      : exprs(args...) {}
  inline generic_op(OperandTypes&& ... args) noexcept
      : exprs(std::move(args)...) {}

  inline result_type eval(eval_context& ctx) const;

 protected:
  mutable std::shared_ptr<std::unique_ptr<result_type>> cache_ptr =
      std::make_shared<std::unique_ptr<result_type>>(nullptr);
};

template <int OpID, template<typename> class ResultExprType, typename ResultType, typename Expr>
using unary_op = generic_op<OpID, ResultExprType, ResultType, Expr>;

template <int OpID, template<typename> class ResultExprType, typename ResultType, typename LhsExpr, typename RhsExpr>
using binary_op = generic_op<OpID, ResultExprType, ResultType, LhsExpr, RhsExpr>;

}  // namespace mgcpp

#endif  // GENERIC_OP_HPP
