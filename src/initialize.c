// #include <type_traits>

#include <cuda_runtime.h>

// #include <mgcpp/cuda/initialize.hpp>

// namespace mg
// {
//     template<typename ElemType,
// 	typename = std::enable_if<std::is_arithmetic<ElemType>::value>>
//     ElemType* cuda_malloc(size_t size)
//     {
//         ElemType* ptr = nullptr;
// 	cudaMalloc((void**)(&ptr), size * sizeof(ElemType));

// 	// if(!ptr)
// 	// {
// 	//     throw;
// 	// }

// 	return ptr;
//     }
// }

void* mg_cuda_malloc(size_t size)
{
    void* ptr = NULL;
    cudaMalloc((void**)(&ptr), size);
}
