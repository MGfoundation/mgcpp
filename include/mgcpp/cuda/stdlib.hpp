#ifndef CUDA_TEMPLATE_STDLIB_HPP
#define CUDA_TEMPLATE_STDLIB_HPP

#include <type_traits>
#include <cstdlib>
#include <new>

namespace mgcpp
{
    
    template<typename ElemType,
             typename = std::enable_if<
                 std::is_arithmetic<ElemType>::value>>
    ElemType* cuda_malloc(size_t size);

    template<typename ElemType,
             typename = std::enable_if<
                 std::is_arithmetic<ElemType>::value>>
    ElemType* cuda_malloc(size_t size,
                          std::nothrow_t const& throw_flag) noexcept;

    template<typename ElemType>
    bool cuda_free(ElemType* ptr) noexcept;
}

#include <mgcpp/cuda/stdlib.tpp>
#endif
