#include <mgcpp/cuda/internal/cuda_stdlib.hpp>
#include <mgcpp/cuda/cuda_template_stdlib.hpp>

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
