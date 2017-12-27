
#include <cstdio>
#include <algorithm>
#include <mgcpp/kernels/bits/fft.cuh>

#define BLK 64Lu
#define PI(T) static_cast<T>(3.141592653589793238462643383279502884197169399375105820974944)

namespace mgcpp
{
    template<typename T>
    __global__  void
    mgblas_Cfft_impl(complex<T> const *x, complex<T> *y, size_t n, size_t m)
    {
        __shared__ complex<T> s[BLK];
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
                    T phi = -2 * PI(T) * i / k;
                    complex<T> z = {cos(phi), sin(phi)};
                    complex<T> u = s[a], v = s[b] * z;
                    s[a] = u + v;
                    s[b] = u - v;
                }
                __syncthreads();
            }
            y[idx] = s[tid];
        }
    }

    template<typename T>
    __global__  void
    mgblas_Cfft_impl2(complex<T> const *x, complex<T> *y, size_t n, size_t level, size_t m)
    {
        __shared__ complex<T> s[BLK];
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
                    T phi = -2 * PI(T) * (sidx % (k * level)) / (k * level);
                    complex<T> z = {cos(phi), sin(phi)}; // z = W_k^(idx%k)
                    complex<T> u = s[a], v = s[b] * z;
                    s[a] = u + v;
                    s[b] = u - v;
                }
                __syncthreads();
            }

            y[sidx] = s[tid];
        }
    }

    kernel_status_t
    mgblas_Cfft(complex<float> const *x, complex<float> *y, size_t n)
    {
        if (n < 1) return invalid_range;

        int grid_size = static_cast<int>(ceil(static_cast<float>(n)/ BLK));
        mgblas_Cfft_impl<float><<<grid_size, BLK>>>(x, y, n, std::min(n, BLK));
        for (size_t m = n / BLK, level = BLK; m > 1; level *= BLK, m /= BLK) {
            mgblas_Cfft_impl2<float><<<grid_size, BLK>>>(y, y, n, level, std::min(m, BLK));
        }
        return success;
    }
}
