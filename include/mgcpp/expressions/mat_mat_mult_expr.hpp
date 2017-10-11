
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_MAT_MAT_MULT_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_MAT_MAT_MULT_EXPR_HPP_

#include <mgcpp/gpu/matrix.hpp>
#include <mgcpp/expressions/mult_expr.hpp>
#include <mgcpp/expressions/base_expr.hpp>
#include <mgcpp/type_traits/gpu_mat.hpp>

namespace mgcpp
{
    template<typename GpuMat>
    struct mult_expr<GpuMat, GpuMat> : public expression
    {
        GpuMat&& _lhs;
        GpuMat&& _rhs;

        inline mult_expr(GpuMat&& lhs, GpuMat&& rhs) noexcept;

        inline typename std::decay<GpuMat>::type
        eval();
    };

    template<typename GpuMat>
    using mat_mat_mult_expr = mult_expr<GpuMat, GpuMat>;

    namespace gpu
    {
        template<typename GpuMat, typename =
                 typename assert_gpu_matrix<GpuMat>::result>
        inline mat_mat_mult_expr<GpuMat> 
        operator*(GpuMat&& lhs, GpuMat&& rhs);
    }
}

#include <mgcpp/expressions/mat_mat_mult_expr.tpp>
#endif
