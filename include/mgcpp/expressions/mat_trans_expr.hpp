
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_MAT_TRANS_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_MAT_TRANS_EXPR_HPP_

#include <mgcpp/gpu/forward.hpp>

namespace mgcpp
{
    template<typename GpuMat>
    struct trans_expr<GpuMat>
    {
        using result_type = typename std::decay<GpuMat>::type;

        GpuMat&& _mat;

        inline trans_expr(GpuMat&& mat) noexcept;

        inline result_type
        eval();
    };

    template<typename GpuMat>
    using mat_trans_expr = trans_expr<GpuMat>;

    template<typename GpuMat,
             MGCPP_CONCEPT(assert_gpu_matrix<GpuMat>)>
    inline mat_trans_expr<GpuMat>
    trans(GpuMat&& mat) noexcept;
}

#endif
