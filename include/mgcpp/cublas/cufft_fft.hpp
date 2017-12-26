
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
    template<typename T>
    inline outcome::result<void>
    cublas_rfft(size_t n, T const* x, complex<T>* result);

    /** Performs complex-to-real inverse unnormalized FFT.
     *  \param n fft size
     *  \param x input array of floor(n/2)+1 interleaved complex values
     *  \param result the fft result, which is an array of n real numbers.
     */
    template<typename T>
    inline outcome::result<void>
    cublas_irfft(size_t n, complex<T> const* x, T* result);

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
    template<typename T>
    inline outcome::result<void>
    cublas_cfft(size_t n, complex<T> const* x, complex<T>* result, cublas::fft_direction direction);
}

#include <mgcpp/cublas/cufft_fft.tpp>

#endif
