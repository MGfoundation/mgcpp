
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/kernels/bits/hadamard.cuh>
#include <cmath>

#define BLK 64

namespace mgcpp
{
    __global__ void
    mgblas_Svhp_impl(float const* x, float const* y,
		     float* z, size_t size)
    {
	int const id = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float shared_x[BLK];
	__shared__ float shared_y[BLK];
	__shared__ float shared_z[BLK];

	shared_x[threadIdx.x] = x[id];
	shared_y[threadIdx.x] = y[id];
	shared_z[threadIdx.x] = z[id];
	__syncthreads();

	if(id < size)
	    shared_z[threadIdx.x] = __fmul_rn(shared_x[threadIdx.x], shared_y[threadIdx.x]);
	__syncthreads();

	z[id] = shared_z[threadIdx.x];
    }

    __global__ void
    mgblas_Dvhp_impl(double const* x, double const* y,
		     double* z, size_t size)
    {
	int const id = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ double shared_x[BLK];
	__shared__ double shared_y[BLK];
	__shared__ double shared_z[BLK];

	shared_x[threadIdx.x] = x[id];
	shared_y[threadIdx.x] = y[id];
	shared_z[threadIdx.x] = z[id];
	__syncthreads();

	if(id < size)
	    shared_z[threadIdx.x] = __fmul_rn(shared_x[threadIdx.x], shared_y[threadIdx.x]);
	__syncthreads();

	z[id] = shared_z[threadIdx.x];
    }

    __global__ void
    mgblas_Hvhp_impl(__half const* x, __half const* y,
		     __half* z, size_t size)
    {
	int const id = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ __half shared_x[BLK];
	__shared__ __half shared_y[BLK];
	__shared__ __half shared_z[BLK];

	shared_x[threadIdx.x] = x[id];
	shared_y[threadIdx.x] = y[id];
	shared_z[threadIdx.x] = z[id];
	__syncthreads();

	if(id < size)
	    shared_z[threadIdx.x] = __fmul_rn(shared_x[threadIdx.x], shared_y[threadIdx.x]);
	__syncthreads();

	z[id] = shared_z[threadIdx.x];
    }

    kernel_status_t
    mgblas_Svhp(float const* x, float const* y,
		float* z, size_t size)
    {
	if(size == 0)
	    return invalid_range;

	int grid_size =
	    static_cast<int>(
		ceil(static_cast<float>(size)/ BLK ));
	mgblas_Svhp_impl<<<BLK, grid_size>>>(x, y, z, size);

	return success;
    }

    kernel_status_t
    mgblas_Dvhp(double const* x, double const* y,
		double* z, size_t size)
    {
	if(size == 0)
	    return invalid_range;
	
	int grid_size =
	    static_cast<int>(
		ceil(static_cast<float>(size)/ BLK ));
	mgblas_Dvhp_impl<<<BLK, grid_size>>>(x, y, z, size);

	return success;
    }

    kernel_status_t
    mgblas_Hvhp(__half const* x, __half const* y,
		__half* z, size_t size)
    {
	if(size == 0)
	    return invalid_range;
	
	int grid_size =
	    static_cast<int>(
		ceil(static_cast<float>(size)/ BLK ));
	mgblas_Hvhp_impl<<<BLK, grid_size>>>(x, y, z, size);

	return success;
    }
}