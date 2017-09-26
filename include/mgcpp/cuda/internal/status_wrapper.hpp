
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CUDA_STATUS_WRAPPER_HPP_
#define _MGCPP_CUDA_STATUS_WRAPPER_HPP_

#include <cstdlib>

#include <mgcpp/cuda/internal/cuda_error.hpp>

namespace mgcpp
{
    namespace internal
    {
        cuda_error_t cuda_mem_get_info(size_t* free, size_t* total);
    }
}

#endif
