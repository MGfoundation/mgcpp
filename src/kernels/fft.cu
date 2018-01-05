
#include <cstdio>
#include <algorithm>

#include <mgcpp/kernels/bits/fft.cuh>
#include <cuComplex.h>

#define BLK 64Lu

#define PI(T) static_cast<T>(3.141592653589793238462643383279502884197169399375105820974944)

namespace mgcpp
{
    __global__  void
    mgblas_Cfft_impl(cuComplex const *x, cuComplex *y, size_t n, size_t m, int dir)
    {
        __shared__ cuComplex s[BLK];
        int const tid = threadIdx.x;
        int const idx = blockIdx.x * BLK + tid;

        if (idx < n) {
            int const ridx = __brev(idx) >> (__clz(n) + 1);
            s[tid] = x[ridx];
            __syncthreads();
            for (int k = 2; k <= m; k <<= 1) {
                int const i = tid % k;
                if (i < k / 2) {
                    int const a = tid;
                    int const b = a + k / 2;
                    float phi = dir * 2 * PI(float) * i / k;
                    cuComplex z = make_cuComplex(cos(phi), sin(phi));
                    cuComplex u = s[a], v = cuCmulf(s[b], z);
                    s[a] = cuCaddf(u, v);
                    s[b] = cuCsubf(u, v);
                }
                __syncthreads();
            }
            y[idx] = s[tid];
        }
    }

    __global__  void
    mgblas_Cfft_impl2(cuComplex const *x, cuComplex *y, size_t n, size_t level, size_t m, int dir)
    {
        __shared__ cuComplex s[BLK];
        int const tid = threadIdx.x;
        int const idx = blockIdx.x * BLK + tid;
        int const jump = n / level;
        int const sidx = idx / jump + (idx % jump) * level;

        if (sidx < n) {
            s[tid] = x[sidx];
            __syncthreads();
            for (int k = 2; k <= m; k <<= 1) {
                int const i = tid % k;
                if (i < k / 2) {
                    int const a = tid;
                    int const b = a + k / 2;
                    int const j = sidx % (k * level);
                    float phi = dir * 2 * PI(float) * j / (k * level);
                    cuComplex z = make_cuComplex(cos(phi), sin(phi)); // z = W_k^(idx%k)
                    cuComplex u = s[a], v = cuCmulf(s[b], z);
                    s[a] = cuCaddf(u, v);
                    s[b] = cuCsubf(u, v);
                }
                __syncthreads();
            }

            y[sidx] = s[tid];
        }
    }

    mgblas_error_t
    mgblas_Cfft(cuComplex const *x, cuComplex *y, size_t n, bool is_inv)
    {
        if (n < 1) return invalid_range;
        int grid_size = static_cast<int>(ceil(static_cast<float>(n)/ BLK));

        int dir = is_inv? 1 : -1;
        mgblas_Cfft_impl<<<grid_size, BLK>>>(x, y, n, std::min(n, BLK), dir);
        for (size_t m = n / BLK, level = BLK; m > 1; level *= BLK, m /= BLK) {
            mgblas_Cfft_impl2<<<grid_size, BLK>>>(y, y, n, level, std::min(m, BLK), dir);
        }
        return success;
    }
}
