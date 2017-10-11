
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/mat_mat_mult_expr.hpp>
#include <mgcpp/operations/mult.hpp>

namespace mgcpp
{
    template<typename GpuMat>
    mult_expr<GpuMat, GpuMat>::
    mult_expr(GpuMat&& lhs, GpuMat&& rhs) noexcept
        :_lhs(std::forward<GpuMat>(lhs)),
         _rhs(std::forward<GpuMat>(rhs)) {}

    template<typename GpuMat>
    typename std::decay<GpuMat>::type
    mult_expr<GpuMat, GpuMat>::
    eval()
    {
        return mult(_lhs, _rhs);
    }

    template<typename GpuMat, typename>
    mat_mat_mult_expr<GpuMat>
    gpu::
    operator*(GpuMat&& lhs,  GpuMat&& rhs)
    {
        return mult_expr<GpuMat, GpuMat>(
            std::forward<GpuMat>(lhs), std::forward<GpuMat>(rhs));
    }
}
