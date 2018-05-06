#ifndef PLACEHOLDER_HPP
#define PLACEHOLDER_HPP

#include <mgcpp/expressions/forward.hpp>

namespace mgcpp {
template <int PlaceholderID,
          template <typename> class ResultExprType,
          typename ResultType>
using placeholder_node =
    generic_expr<int, PlaceholderID, ResultExprType, ResultType, 0>;
}

#endif  // PLACEHOLDER_HPP
