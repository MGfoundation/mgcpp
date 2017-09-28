#ifndef _MGCPP_CUDA_MEMORY_MANAGEMENT_HPP_
#define _MGCPP_CUDA_MEMORY_MANAGEMENT_HPP_

#include <cstdlib>

#include <cuda_runtime.h>

#include <outcome.hpp>

namespace outcome = OUTCOME_V2_NAMESPACE;

namespace mgcpp
{
    using free_mem_t = size_t;
    using total_mem_t = size_t;

    outcome::result<std::pair<free_mem_t, total_mem_t>>
    cuda_mem_get_info() noexcept;
}

#endif
