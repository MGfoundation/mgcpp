
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_KERNELS_MGBLAS_LV1_HPP_
#define _MGCPP_KERNELS_MGBLAS_LV1_HPP_

#include <outcome.hpp>
namespace outcome = OUTCOME_V2_NAMESPACE;

#include <cstdlib>

namespace mgcpp
{
    template<typename T>
    inline outcome::result<void>
    mgblas_vhp(T const* x, T const* y, T* z, size_t n);
}

#include <mgcpp/kernels/mgblas_lv1.tpp>
#endif
