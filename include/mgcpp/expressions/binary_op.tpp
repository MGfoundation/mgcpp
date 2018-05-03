#ifndef BINARY_OP_TPP
#define BINARY_OP_TPP

#include <mgcpp/expressions/binary_op.hpp>
#include <mgcpp/expressions/evaluator.hpp>

#include <cstdio>

namespace mgcpp {

template <int OpID, typename LhsExpr, typename RhsExpr, template<typename> class ResultExprType, typename ResultType>
typename binary_op<OpID, LhsExpr, RhsExpr, ResultExprType, ResultType>::result_type
binary_op<OpID, LhsExpr, RhsExpr, ResultExprType, ResultType>::eval() const
{
    if (cache_ptr.use_count() > 1) {
        if (*cache_ptr == nullptr)
            *cache_ptr = std::make_unique<result_type>(evaluator::eval(*this));
        return **cache_ptr;
    }
    else {
        return evaluator::eval(*this);
    }
}

}

#endif // BINARY_OP_TPP
