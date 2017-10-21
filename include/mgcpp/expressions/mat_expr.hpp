
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_MAT_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_MAT_EXPR_HPP_

#include <type_traits>

#include <mgcpp/gpu/forward.hpp>
#include <mgcpp/system/concept.hpp>
#include <mgcpp/type_traits/gpu_mat.hpp>
#include <mgcpp/expressions/result_type.hpp>

namespace mgcpp
{
    template<typename GpuMat,
             MGCPP_CONCEPT(assert_gpu_matrix<GpuMat>)>
    inline GpuMat 
    eval(GpuMat&& mat);

    template<typename GpuMat>
    struct result_type<GpuMat,
                       typename assert_gpu_matrix<GpuMat>::result>
    {
        using type = typename std::decay<GpuMat>::type;
    };
}

#include <mgcpp/expressions/mat_expr.tpp>
#endif
