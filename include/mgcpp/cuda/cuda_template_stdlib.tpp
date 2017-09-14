#include <type_traits>

#include <mgcpp/internal_cuda/cuda_stdlib.hpp>

namespace mgcpp
{
    template<typename ElemType,
	typename = std::enable_if<std::is_arithmetic<ElemType>::value>>
    ElemType* cuda_malloc(size_t size)
    {
        void* ptr = nullptr;
        internal::cuda_malloc(ptr, size * sizeof(ElemType));

        // if(!ptr)
	// {
	//     throw;
	// }
        return static_cast<ElemType*>(ptr);
    }
}
