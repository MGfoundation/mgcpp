#ifndef CUDA_STDLIB_HPP
#define CUDA_STDLIB_HPP

#include <stdlib.h>

namespace mgcpp
{
    namespace internal
    {
        enum cuda_error_t
        {
            cuda_success = 0,
            cuda_error_memory_allocation = 2,
            cuda_error_initialization_error = 3,
            cuda_error_invalid_device_pointer = 17
        };

        cuda_error_t cuda_malloc(void* ptr, size_t size);
        cuda_error_t cuda_free(void* ptr);

        void some_shit();

    }
}

#endif
