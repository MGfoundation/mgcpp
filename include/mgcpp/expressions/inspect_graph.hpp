#ifndef INSPECT_GRAPH_HPP
#define INSPECT_GRAPH_HPP

#include <ostream>

#include <mgcpp/expressions/expression.hpp>

namespace mgcpp {

template <typename Expr>
std::ostream& operator<<(std::ostream& os, expression<Expr> const& expr);
}

#include <mgcpp/expressions/inspect_graph.tpp>
#endif  // INSPECT_GRAPH_HPP
