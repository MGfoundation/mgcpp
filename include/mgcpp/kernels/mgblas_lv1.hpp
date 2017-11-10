
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_KERNELS_MGBLAS_LV1_HPP_
#define _MGCPP_KERNELS_MGBLAS_LV1_HPP_

#include <boost/outcome.hpp>
namespace outcome = BOOST_OUTCOME_V2_NAMESPACE;

#include <cstdlib>

namespace mgcpp
{
    template<typename T>
    inline outcome::result<void>
    mgblas_vhp(T const* x, T const* y, T* z, size_t n);

    template<typename T>
    inline outcome::result<void>
    mgblas_vab(T* x, size_t n);

    template<typename T>
    inline outcome::result<void>
    mgblas_vpr(T const* x, T* y, size_t n);
}

#include <mgcpp/kernels/mgblas_lv1.tpp>
#endif
