
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/type_traits/mat_mat_expr.hpp>

#include <mgcpp/gpu/forward.hpp>
#include <mgcpp/expressions/expression_base.hpp>

namespace mgcpp
{
    template<typename Rhs, typename Lhs>
    // typename assert_same_gpu_matrix<Rhs, Lhs>::result>
    struct mat_mat_mult_expr
        : expression
    {
        RhsType& lhs;
        LhsType& rhs;

        mat_mat_mult_expr(RhsType& lhs, LhsType& rhs);
    };

    template<typename LhsMat, typename RhsMat,
             typename assert_same_gpu_matrix<RhsMat,
                                             LhsMat>::result>
    mat_mat_mult_expr<LhsMat, RhsMat>
    operator*(LhsMat& lhs,  RhsMat& rhs);
}
