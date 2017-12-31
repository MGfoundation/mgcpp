
#ifndef _MGCPP_BLAS_FFT_HPP_
#define _MGCPP_BLAS_FFT_HPP_

#include <mgcpp/global/complex.hpp>

#include <boost/outcome.hpp>
namespace outcome = BOOST_OUTCOME_V2_NAMESPACE;

#include <cufft.h>

namespace mgcpp
{
    /** Performs real-to-complex forward FFT.
     *  \param n fft size
     *  \param x input array of n real values
     *  \param result the fft result, which is an array of
     *       interleaved floor(n/2)+1 complex numbers.
     */
    inline outcome::result<void>
    cublas_rfft(size_t n, float const* x, cuComplex* result);
    inline outcome::result<void>
    cublas_rfft(size_t n, double const* x, cuDoubleComplex* result);

    /** Performs complex-to-real inverse unnormalized FFT.
     *  \param n fft size
     *  \param x input array of floor(n/2)+1 interleaved complex values
     *  \param result the fft result, which is an array of n real numbers.
     */
    inline outcome::result<void>
    cublas_irfft(size_t n, cuComplex const* x, float* result);
    inline outcome::result<void>
    cublas_irfft(size_t n, cuDoubleComplex const* x, double* result);

    namespace cublas
    {
        enum class fft_direction
        {
            forward = CUFFT_FORWARD,
            inverse = CUFFT_INVERSE
        };
    }

    /** Performs complex-to-complex FFT.
     *  \param n fft size
     *  \param x input array of n interleaved complex values
     *  \param result the fft result, which is
     *      an array of n interleaved complex values.
     */
    inline outcome::result<void>
    cublas_cfft(size_t n, size_t m, cuComplex const* x, cuComplex* result, cublas::fft_direction direction);
    inline outcome::result<void>
    cublas_cfft(size_t n, size_t m, cuDoubleComplex const* x, cuDoubleComplex* result, cublas::fft_direction direction);

    inline outcome::result<void>
    cublas_rfft2(size_t n, size_t m, float const* x, cuComplex* result);
    inline outcome::result<void>
    cublas_rfft2(size_t n, size_t m, double const* x, cuDoubleComplex* result);

    inline outcome::result<void>
    cublas_irfft2(size_t n, size_t m, cuComplex const* x, float* result);
    inline outcome::result<void>
    cublas_irfft2(size_t n, size_t m, cuDoubleComplex const* x, double* result);

    inline outcome::result<void>
    cublas_cfft2(size_t n, size_t m, cuComplex const* x, cuComplex* result, cublas::fft_direction direction);
    inline outcome::result<void>
    cublas_cfft2(size_t n, size_t m, cuDoubleComplex const* x, cuDoubleComplex* result, cublas::fft_direction direction);
}

#include <mgcpp/cublas/cufft_fft.tpp>

#endif
