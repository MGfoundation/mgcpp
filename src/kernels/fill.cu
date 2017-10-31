#include <mgcpp/kernels/bits/fill.cuh>

#include <stdint.h>

namespace mgcpp
{
    void fill64(void* ptr, int bitvec, size_t size)
    {
	
    }

    // void fill32(void* ptr, uint32_t bitvec, size_t size)
    // {
	
    // }

    // void fill16(void* ptr, uint16_t bitvec, size_t size)
    // {
	
    // }

    __host__ __device__  void
    fill64_impl(int* ptr, int bitvec, size_t count)
    {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (size_t i = idx;
	     i < count;
	     i += gridDim.x * blockDim.x)
	{
	    ptr[i] = bitvec;
	}
    }

    // __host__ __device__  void
    // fill32_impl(uint32_t* ptr, uint32_t bitvec, size_t count)
    // {
    // 	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 	for (size_t i = idx;
    // 	     i < count;
    // 	     i += gridDim.x * blockDim.x)
    // 	{
    // 	    ptr[i] = bitvec;
    // 	}
    // }

    // __host__ __device__  void
    // fill16_impl(uint16_t* ptr, uint16_t bitvec, size_t count)
    // {
    // 	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 	for (size_t i = idx;
    // 	     i < count;
    // 	     i += gridDim.x * blockDim.x)
    // 	{
    // 	    ptr[i] = bitvec;
    // 	}
    // }
}