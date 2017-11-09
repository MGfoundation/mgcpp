
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_KERNELS_MGBLAS_HELPERS_HPP_
#define _MGCPP_KERNELS_MGBLAS_HELPERS_HPP_

#include <mgcpp/kernels/bits/fill.cuh>

#include <boost/outcome.hpp>
namespace outcome = BOOST_OUTCOME_V2_NAMESPACE;

#include <cstdlib>

namespace mgcpp
{
    template<typename T>
    inline outcome::result<void>
    mgblas_fill(T* arr, T value, size_t n);
}

#include <mgcpp/kernels/mgblas_helpers.tpp>
#endif
