
#include <mgcpp/kernels/bits/fft.cuh>

//#define BLK 8
#define BLK 64
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
    mgblas_rfft_impl(T const *x, cmplx<T> *y, size_t n)
    {
        __shared__ cmplx<T> s[BLK];
        int const idx = blockIdx.x * blockDim.x + threadIdx.x;
        int const tid = threadIdx.x;

        if (idx < n) {
            s[tid].real = x[idx];
            s[tid].imag = 0;
            __syncthreads();

            for (int k = 2; k <= n; k <<= 1) {
                    int const i = idx % k;
                    if (i < k / 2) {
                        int const a = idx - blockIdx.x * blockDim.x;
                        int const b = a + k / 2;
                        T phi = -2 * PI(T) * i / k;
                        cmplx<T> z = {cos(phi), sin(phi)}; // z = W_k^(idx%k)
                        cmplx<T> u = s[a];
                        s[a] = u + s[b] * z;
                        s[b] = u - s[b] * z;
                    }
                __syncthreads();
            }

            y[idx] = s[tid];
        }
    }

    kernel_status_t
    mgblas_Srfft(float const *x, float *y, size_t n)
    {
        cmplx<float> *cy = reinterpret_cast<cmplx<float>*>(y);
	int grid_size = static_cast<int>(ceil(static_cast<float>(n)/ BLK));
        mgblas_rfft_impl<float><<<grid_size, BLK>>>(x, cy, n);

        return success;
    }
}
