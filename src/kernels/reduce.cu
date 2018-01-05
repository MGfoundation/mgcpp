
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/kernels/bits/hadamard.cuh>

#include <cmath>
#include <stdint.h>

#define BLK 64

namespace mgcpp
{
    __device__ double double_cas_add(double* address, double val)
    {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
    }

    __global__ void
    mgblas_Svpr_impl(float const* x, float* y, size_t n)
    {
        __shared__ float shared[BLK];

        size_t const tid = threadIdx.x;
        size_t const id = blockIdx.x * blockDim.x + threadIdx.x;

        if(id >= n)
            return;

        shared[tid] = x[id];
        __syncthreads();

        if(blockIdx.x == gridDim.x - 1)
        {
            atomicAdd(y, shared[tid]);
            return;
        }
        __syncthreads();

        for(size_t stride = blockDim.x / 2; stride > 0u; stride >>= 1)
        {
            if(tid < stride)
                shared[tid] += shared[tid + stride];
            __syncthreads();
        }
              
        if(tid == 0u)
            atomicAdd(y, shared[0]); 
    }

    __global__ void
    mgblas_Dvpr_impl(double const* x, double* y, size_t n)
    {
        __shared__ double shared[BLK];

        size_t const tid = threadIdx.x;
        size_t const id = blockIdx.x * blockDim.x + threadIdx.x;

        if(id >= n)
            return;

        shared[tid] = x[id];
        __syncthreads();

        if(blockIdx.x == gridDim.x - 1)
        {
#if __CUDA_ARCH__ < 600
            double_cas_add(y, shared[tid]);
#else
            atomicAdd(y, shared[tid]);
#endif
            return;
        }
        __syncthreads();

        for(size_t stride = blockDim.x / 2; stride > 0u; stride >>= 1)
        {
            if(tid < stride)
                shared[tid] += shared[tid + stride];
            __syncthreads();
        }
              
        if(tid == 0u)
        {
#if __CUDA_ARCH__ < 600
            double_cas_add(y, shared[0]);
#else
            atomicAdd(y, shared[0]);
#endif
        }
    }

    // __global__ void
    // mgblas_Hvhp_impl(__half const* x, __half const* y,
    // 		     __half* z, size_t size)
    // {
    // 	int const id = blockIdx.x * blockDim.x + threadIdx.x;

    // 	__shared__ __half shared_x[BLK];
    // 	__shared__ __half shared_y[BLK];
    // 	__shared__ __half shared_z[BLK];

    // 	shared_x[threadIdx.x] = x[id];
    // 	shared_y[threadIdx.x] = y[id];
    // 	shared_z[threadIdx.x] = z[id];
    // 	__syncthreads();

    // 	if(id < size)
    // 	    shared_z[threadIdx.x] = __hmul(shared_x[threadIdx.x], shared_y[threadIdx.x]);
    // 	__syncthreads();

    // 	z[id] = shared_z[threadIdx.x];
    // }

    mgblas_error_t
    mgblas_Svpr(float const* x, float* y, size_t size)
    {
        if(size == 0)
            return invalid_range;

        float* result;
        cudaError_t alloc_status =
            cudaMalloc((void**)&result, sizeof(float));

        int grid_size =
            static_cast<int>(
                ceil(static_cast<float>(size)/ BLK ));
        mgblas_Svpr_impl<<<BLK, grid_size>>>(x, result, size);

        cudaError_t copy_status =
            cudaMemcpy(y, result, sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(result);
        return success;
    }

    mgblas_error_t
    mgblas_Dvpr(double const* x, double* y, size_t size)
    {
        if(size == 0)
            return invalid_range;

        double* result;
        cudaError_t alloc_status =
            cudaMalloc((void**)&result, sizeof(double));

        int grid_size =
            static_cast<int>(
                ceil(static_cast<float>(size)/ BLK ));
        mgblas_Dvpr_impl<<<BLK, grid_size>>>(x, result, size);

        cudaError_t copy_status =
            cudaMemcpy(y, result, sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(result);
        return success;
    }

    // kernel_status_t
    // mgblas_Hvhp(__half const* x, __half const* y,
    // 		__half* z, size_t size)
    // {
    // 	if(size == 0)
    // 	    return invalid_range;
	
    // 	int grid_size =
    // 	    static_cast<int>(
    // 		ceil(static_cast<float>(size)/ BLK ));
    // 	mgblas_Hvhp_impl<<<BLK, grid_size>>>(x, y, z, size);

    // 	return success;
    // }
}