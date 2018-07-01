
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/kernels/bits/hadamard.cuh>

#include <cmath>
#include <cstdio>
#include <stdint.h>

#define BLK 128
#define THRES 1024

namespace mgcpp
{
    // __device__ double double_cas_add(double* address, double val)
    // {
    //     unsigned long long int* address_as_ull = (unsigned long long int*)address;
    //     unsigned long long int old = *address_as_ull, assumed;
    //     do {
    //         assumed = old;
    //         old = atomicCAS(address_as_ull, assumed,
    //                         __double_as_longlong(val + __longlong_as_double(assumed)));
    //     } while (assumed != old);
    //     return __longlong_as_double(old);
    // }

    template<typename It, typename Type>
    inline Type kahan_reduce(It begin, It end, Type init)
    {
        Type sum = init;
        Type running_error = Type();
        Type temp;
        Type difference;

        for (; begin != end; ++begin) {
            difference = *begin;
            difference -= running_error;
            temp = sum;
            temp += difference;
            running_error = temp;
            running_error -= sum;
            running_error -= difference;
            sum = temp;
        }
        return sum;
    }

#ifdef USE_HALF
    inline __half kahan_reduce(__half* begin,
                               __half* end,
                               float init)
    {
        float sum = init;
        float running_error = 0.0f;
        float temp;
        float difference;

        for (; begin != end; ++begin) {
            difference = __half2float(*begin);
            difference -= running_error;
            temp = sum;
            temp += difference;
            running_error = temp;
            running_error -= sum;
            running_error -= difference;
            sum = temp;
        }
        return __float2half(sum);
    }
#endif

    __global__ void
    mgblas_Svpr_impl(float const* x, float* y, size_t n)
    {
        __shared__ float shared[BLK];

        size_t const bid = blockIdx.x;
        size_t const tid = threadIdx.x;
        size_t const id = blockIdx.x * blockDim.x + threadIdx.x;

        if(id >= n)
            shared[tid] = 0;
        else
            shared[tid] = x[id];
        __syncthreads();

        for(size_t stride = blockDim.x / 2; stride > 0u; stride >>= 1)
        {
            if(tid < stride)
                shared[tid] += shared[tid + stride];
            __syncthreads();
        }
              
        if(tid == 0u)
            y[bid] = shared[0]; 
    }

    __global__ void
    mgblas_Dvpr_impl(double const* x, double* y, size_t n)
    {
        __shared__ double shared[BLK];

        size_t const bid = blockIdx.x;
        size_t const tid = threadIdx.x;
        size_t const id = blockIdx.x * blockDim.x + threadIdx.x;

        if(id >= n)
            shared[tid] = 0;
        else
            shared[tid] = x[id];
        __syncthreads();

        for(size_t stride = blockDim.x / 2; stride > 0u; stride >>= 1)
        {
            if(tid < stride)
                shared[tid] += shared[tid + stride];
            __syncthreads();
        }
              
        if(tid == 0u)
            y[bid] = shared[0]; 
    }

#ifdef USE_HALF
    __global__ void
    mgblas_Hvpr_impl(__half const* x, __half* y, size_t n)
    {
        __shared__ __half shared[BLK];

        size_t const bid = blockIdx.x;
        size_t const tid = threadIdx.x;
        size_t const id = blockIdx.x * blockDim.x + threadIdx.x;

        if(id >= n)
            shared[tid] = __float2half(0.0f);
        else
            shared[tid] = x[id];
        __syncthreads();

        for(size_t stride = blockDim.x / 2; stride > 0u; stride >>= 1)
        {
            if(tid < stride)
                shared[tid] = __hadd(shared[tid], shared[tid + stride])
                    __syncthreads();
        }
              
        if(tid == 0u)
            y[bid] = shared[0]; 
    }
#endif

    mgblas_error_t
    mgblas_Svpr(float const* x, float* y, size_t size)
    {
        if(size == 0)
            return invalid_range;


        int grid_size =
            static_cast<int>(
                ceil(static_cast<float>(size)/ BLK ));

        float host_buffer[THRES];
        float* device_buffer; 
        cudaError_t alloc_status =
            cudaMalloc((void**)&device_buffer, sizeof(float) * grid_size);
        if(alloc_status != cudaSuccess)
            return memory_allocation_failure;

        mgblas_Svpr_impl<<<grid_size, BLK>>>(x, device_buffer, size);

        if(grid_size >= THRES)
        {
            grid_size = static_cast<int>(
            ceil(static_cast<float>(grid_size)/ BLK ));
            mgblas_Svpr_impl<<<grid_size, BLK>>>(device_buffer, device_buffer, size);
        }

        cudaError_t copy_status =
            cudaMemcpy((void*)host_buffer,
                       (void*)device_buffer,
                       sizeof(float) * grid_size,
                       cudaMemcpyDeviceToHost);

        if(copy_status != cudaSuccess)
            return device_to_host_memcpy_failure;

        *y = kahan_reduce(&host_buffer[0], &host_buffer[grid_size], 0.0);

        cudaFree(device_buffer);
        return success;
    }

    mgblas_error_t
    mgblas_Dvpr(double const* x, double* y, size_t size)
    {
        if(size == 0)
            return invalid_range;

        int grid_size =
            static_cast<int>(
                ceil(static_cast<float>(size)/ BLK ));

        double host_buffer[THRES];
        double* device_buffer; 
        cudaError_t alloc_status =
            cudaMalloc((void**)&device_buffer, sizeof(double) * grid_size);
        if(alloc_status != cudaSuccess)
            return memory_allocation_failure;

        mgblas_Dvpr_impl<<<grid_size, BLK>>>(x, device_buffer, size);

        if(grid_size >= THRES)
        {
            grid_size = static_cast<int>(
                ceil(static_cast<float>(grid_size)/ BLK ));
            mgblas_Dvpr_impl<<<grid_size, BLK>>>(device_buffer, device_buffer, size);
        }

        cudaError_t copy_status =
            cudaMemcpy((void*)host_buffer,
                       (void*)device_buffer,
                       sizeof(double) * grid_size,
                       cudaMemcpyDeviceToHost);

        if(copy_status != cudaSuccess)
            return device_to_host_memcpy_failure;

        *y = kahan_reduce(&host_buffer[0], &host_buffer[grid_size], 0.0);

        cudaFree(device_buffer);
        return success;
    }

#ifdef USE_HALF
    mgblas_error_t
    mgblas_Hvpr(__half const* x, __half* y, size_t size)
    {
        if(size == 0)
            return invalid_range;

        int grid_size =
            static_cast<int>(
                ceil(static_cast<float>(size)/ BLK ));

        __half host_buffer[THRES];
        __half* device_buffer; 
        cudaError_t alloc_status =
            cudaMalloc((void**)&device_buffer, sizeof(__half) * grid_size);
        if(alloc_status != cudaSuccess)
            return memory_allocation_failure;

        mgblas_Hvpr_impl<<<grid_size, BLK>>>(x, device_buffer, size);

        if(grid_size >= THRES)
        {
            grid_size = static_cast<int>(
                ceil(static_cast<float>(grid_size)/ BLK ));
            mgblas_Hvpr_impl<<<grid_size, BLK>>>(device_buffer, device_buffer, size);
        }

        cudaError_t copy_status =
            cudaMemcpy((void*)host_buffer,
                       (void*)device_buffer,
                       sizeof(__half) * grid_size,
                       cudaMemcpyDeviceToHost);

        if(copy_status != cudaSuccess)
            return device_to_host_memcpy_failure;

        *y = kahan_reduce(&host_buffer[0], &host_buffer[grid_size], 0.0f);

        cudaFree(device_buffer);
        return success;
    }
#endif
}
