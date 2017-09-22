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
