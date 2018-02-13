
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CUDA_LIBS_FFT_HPP_
#define _MGCPP_CUDA_LIBS_FFT_HPP_

#include <mgcpp/global/complex.hpp>

#include <boost/outcome.hpp>
namespace outcome = BOOST_OUTCOME_V2_NAMESPACE;

#include <cufft.h>

namespace mgcpp {
namespace cufft {
/**
 *  Performs single-precision real-to-complex forward FFT.
 *  Effectively calls the corresponding CuFFT function.
 *  \param n fft size
 *  \param x input array of n real values
 *  \param result the fft result, which is an array of
 *         floor(n/2)+1 complex numbers.
 */
inline outcome::result<void> rfft(size_t n, float const* x, cuComplex* result);

/**
 *  Performs double-precision real-to-complex forward FFT.
 *  Effectively calls the corresponding CuFFT function.
 *  \param n fft size
 *  \param x input array of n real values
 *  \param result the fft result, which is an array of
 *         floor(n/2)+1 complex numbers.
 */
inline outcome::result<void> rfft(size_t n,
                                  double const* x,
                                  cuDoubleComplex* result);

/**
 *  Performs single-precision complex-to-real inverse unnormalized FFT.
 *  Effectively calls the corresponding CuFFT function.
 *  \param n fft size
 *  \param x input array of floor(n/2)+1 interleaved complex values
 *  \param result the fft result, which is an array of n real numbers.
 */
inline outcome::result<void> irfft(size_t n, cuComplex const* x, float* result);

/**
 *  Performs double-precision complex-to-real inverse unnormalized FFT.
 *  Effectively calls the corresponding CuFFT function.
 *  \param n fft size
 *  \param x input array of floor(n/2)+1 interleaved complex values
 *  \param result the fft result, which is an array of n real numbers.
 */
inline outcome::result<void> irfft(size_t n,
                                   cuDoubleComplex const* x,
                                   double* result);

enum class fft_direction {
  /// Forward FFT (time domain -> frequency domain representation)
  forward = CUFFT_FORWARD,

  /// Inverse FFT (frequency domain -> time domain representation)
  inverse = CUFFT_INVERSE
};

/**
 *  Performs single-precision complex-to-complex FFT.
 *  Effectively calls the corresponding CuFFT function.
 *  \param n fft size
 *  \param x input array of n complex values
 *  \param result the fft result, which is an array of n complex values.
 *  \param direction cufft::fft_direction::forward for forward FFT, or
 * cufft::fft_direction::inverse to perform an inverse (unnormalized) FFT.
 */
inline outcome::result<void> cfft(size_t n,
                                  cuComplex const* x,
                                  cuComplex* result,
                                  fft_direction direction);

/**
 *  Performs double-precision complex-to-complex FFT.
 *  Effectively calls the corresponding CuFFT function.
 *  \param n fft size
 *  \param x input array of n complex values
 *  \param result the fft result, which is an array of n complex values.
 *  \param direction cufft::fft_direction::forward for forward FFT, or
 * cufft::fft_direction::inverse to perform an inverse (unnormalized) FFT.
 */
inline outcome::result<void> cfft(size_t n,
                                  cuDoubleComplex const* x,
                                  cuDoubleComplex* result,
                                  fft_direction direction);

/**
 *  Performs two-dimensional single-precision real-to-complex forward FFT.
 *  Effectively calls the corresponding CuFFT function.
 *  \param n fastest-changing dimension
 *  \param m slowest-changing dimension
 *  \param x input array of (n x m) real values
 *  \param result the fft result, which is an array of floor(n/2)+1 complex
 * numbers.
 */
inline outcome::result<void> rfft2(size_t n,
                                   size_t m,
                                   float const* x,
                                   cuComplex* result);

/**
 *  Performs two-dimensional double-precision real-to-complex forward FFT.
 *  Effectively calls the corresponding CuFFT function.
 *  \param n fastest-changing dimension
 *  \param m slowest-changing dimension
 *  \param x input array of (n x m) real values
 *  \param result the fft result, which is an array of (floor(n/2)+1) x m
 * complex numbers.
 */
inline outcome::result<void> rfft2(size_t n,
                                   size_t m,
                                   double const* x,
                                   cuDoubleComplex* result);

/**
 *  Performs two-dimensional single-precision complex-to-real inverse
 * unnormalized FFT. Effectively calls the corresponding CuFFT function. \param
 * n fastest-changing dimension \param m slowest-changing dimension \param x
 * input array of (floor(n/2)+1) x m interleaved complex values \param result
 * the fft result, which is an array of n x m real numbers.
 */
inline outcome::result<void> irfft2(size_t n,
                                    size_t m,
                                    cuComplex const* x,
                                    float* result);

/**
 *  Performs two-dimensional double-precision complex-to-real inverse
 * unnormalized FFT. Effectively calls the corresponding CuFFT function. \param
 * n fastest-changing dimension \param m slowest-changing dimension \param x
 * input array of (floor(n/2)+1) x m interleaved complex values \param result
 * the fft result, which is an array of n x m real numbers.
 */
inline outcome::result<void> irfft2(size_t n,
                                    size_t m,
                                    cuDoubleComplex const* x,
                                    double* result);

/**
 *  Performs two-dimensional single-precision complex-to-complex FFT.
 *  Effectively calls the corresponding CuFFT function.
 *  \param n fastest-changing dimension
 *  \param m slowest-changing dimension
 *  \param x input array of n x m interleaved complex values
 *  \param result the fft result, which is an array of n x m real numbers.
 *  \param direction cufft::fft_direction::forward for forward FFT, or
 * cufft::fft_direction::inverse to perform an inverse (unnormalized) FFT.
 */
inline outcome::result<void> cfft2(size_t n,
                                   size_t m,
                                   cuComplex const* x,
                                   cuComplex* result,
                                   fft_direction direction);

/**
 *  Performs two-dimensional double-precision complex-to-complex FFT.
 *  Effectively calls the corresponding CuFFT function.
 *  \param n fastest-changing dimension
 *  \param m slowest-changing dimension
 *  \param x input array of n x m interleaved complex values
 *  \param result the fft result, which is an array of n x m real numbers.
 *  \param direction cufft::fft_direction::forward for forward FFT, or
 * cufft::fft_direction::inverse to perform an inverse (unnormalized) FFT.
 */
inline outcome::result<void> cfft2(size_t n,
                                   size_t m,
                                   cuDoubleComplex const* x,
                                   cuDoubleComplex* result,
                                   fft_direction direction);
}  // namespace cufft
}  // namespace mgcpp

#include <mgcpp/cuda_libs/cufft_fft.tpp>

#endif
