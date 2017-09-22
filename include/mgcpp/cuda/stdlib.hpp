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

    template<typename ElemType>
    void cuda_free(ElemType* ptr);

    template<typename ElemType>
    bool cuda_free_nothrow(ElemType* ptr) noexcept;
}

#include <mgcpp/cuda/stdlib.tpp>
#endif
