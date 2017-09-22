#ifndef _STATUS_WRAPPER_HPP_
#define _STATUS_WRAPPER_HPP_

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
