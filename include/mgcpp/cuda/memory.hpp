//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef MGCPP_CUDA_MEMORY_HPP
#define MGCPP_CUDA_MEMORY_HPP

#include <outcome.hpp>

#include <mgcpp/system/error_code.hpp>

#include <type_traits>
#include <cstdlib>
#include <new>

namespace outcome = OUTCOME_V2_NAMESPACE;

namespace mgcpp
{
    
    template<typename ElemType,
             typename = std::enable_if<
                 std::is_arithmetic<ElemType>::value>>
    outcome::result<ElemType*>
    cuda_malloc(size_t size) noexcept;

    template<typename ElemType>
    outcome::result<void>
    cuda_free(ElemType* ptr) noexcept;

    template<typename ElemType>
    outcome::result<void>
    cuda_memset(ElemType* ptr, ElemType value, size_t count) noexcept;

    template<typename ElemType>
    outcome::result<ElemType*>
    malloc_pinned(size_t count) noexcept;

    template<typename ElemType>
    outcome::result<void>
    free_pinned(ElemType* ptr) noexcept;

    enum class cuda_memcpy_kind;

    template<typename ElemType>
    outcome::result<void>
    cuda_memcpy(ElemType* to, ElemType const* from,
                size_t count, cuda_memcpy_kind kind) noexcept;

    using free_mem_t = size_t;
    using total_mem_t = size_t;

    outcome::result<std::pair<free_mem_t, total_mem_t>>
    inline cuda_mem_get_info() noexcept;
}

#include <mgcpp/cuda/memory.tpp>
#endif
