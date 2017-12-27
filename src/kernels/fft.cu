
#include <cstdio>
#include <algorithm>
#include <mgcpp/kernels/bits/fft.cuh>

#define BLK 64Lu
#define PI(T) static_cast<T>(3.141592653589793238462643383279502884197169399375105820974944)

namespace mgcpp
{
    // device complex type
    template<typename T>
    struct cmplx {
        T real, imag;

        __device__
        cmplx operator* (cmplx rhs) const {
            cmplx r = {
                real * rhs.real - imag * rhs.imag,
                real * rhs.imag + imag * rhs.real
            };
            return r;
        }

        __device__
        cmplx operator+ (cmplx rhs) const {
            cmplx r = {
                real + rhs.real,
                imag + rhs.imag
            };
            return r;
        }

        __device__
        cmplx operator- (cmplx rhs) const {
            cmplx r = {
                real - rhs.real,
                imag - rhs.imag
            };
            return r;
        }
    };

    template<typename T>
    __global__  void
    mgblas_Cfft_impl(cmplx<T> const *x, cmplx<T> *y, size_t n, size_t m)
    {
        __shared__ cmplx<T> s[BLK];
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
                    cmplx<T> z = {cos(phi), sin(phi)};
                    cmplx<T> u = s[a], v = s[b] * z;
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
    mgblas_Cfft_impl2(cmplx<T> const *x, cmplx<T> *y, size_t n, size_t level, size_t m)
    {
        __shared__ cmplx<T> s[BLK];
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
                    cmplx<T> z = {cos(phi), sin(phi)}; // z = W_k^(idx%k)
                    cmplx<T> u = s[a], v = s[b] * z;
                    s[a] = u + v;
                    s[b] = u - v;
                }
                __syncthreads();
            }

            y[sidx] = s[tid];
        }
    }

    kernel_status_t
    mgblas_Cfft(float const *x, float *y, size_t n)
    {
        if (n < 1) return invalid_range;

        cmplx<float> const *cx = reinterpret_cast<cmplx<float> const*>(x);
        cmplx<float> *cy = reinterpret_cast<cmplx<float>*>(y);

        int grid_size = static_cast<int>(ceil(static_cast<float>(n)/ BLK));
        mgblas_Cfft_impl<float><<<grid_size, BLK>>>(cx, cy, n, std::min(n, BLK));
        for (size_t m = n / BLK, level = BLK; m > 1; level *= BLK, m /= BLK) {
            mgblas_Cfft_impl2<float><<<grid_size, BLK>>>(cy, cy, n, level, std::min(m, BLK));
        }
        return success;
    }
}
