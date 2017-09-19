#ifndef CUDA_TEMPLATE_STDLIB_HPP
#define CUDA_TEMPLATE_STDLIB_HPP

#include <type_traits>
#include <cstdlib>

namespace mgcpp
{
    
    template<typename ElemType,
             typename = std::enable_if<
                 std::is_arithmetic<ElemType>::value>>
    ElemType* cuda_malloc(size_t size);

    template<typename ElemType,
             typename = std::enable_if<
                 std::is_arithmetic<ElemType>::value>>
    ElemType* cuda_malloc_nothrow(size_t size) noexcept;
}

#include "cuda_template_stdlib.tpp"

#endif