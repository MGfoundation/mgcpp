
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef CUDA_STDLIB_HPP
#define CUDA_STDLIB_HPP

#include <stdlib.h>
#include <mgcpp/cuda/internal/cuda_error.hpp>

namespace mgcpp
{
    namespace internal
    {
        cuda_error_t cuda_malloc(void** ptr, size_t size);
        cuda_error_t cuda_free(void* ptr);
    }
}

#endif
