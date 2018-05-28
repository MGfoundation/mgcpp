
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_FFT_HPP_
#define _MGCPP_OPERATIONS_FFT_HPP_

#include <mgcpp/global/complex.hpp>
#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

namespace mgcpp {
/// FFT direction
enum class fft_direction {
  /// Forward FFT.
  forward,
  /// Inverse FFT.
  inverse
};

namespace strict {
/**
 *  Performs real-to-complex forward FFT.
 *  \param vec the vector to perform the fft on.
 *  \returns the FFT result.
 */
template <typename DeviceVec, typename Type>
inline decltype(auto) rfft(dense_vector<DeviceVec, Type> const& vec);

/**
 *  Performs complex-to-real inverse normalized FFT.
 *  \param vec the vector to perform the fft on.
 *  \param n performs `((vec.size() - 1) * 2)` point inverse FFT if `n` equals
 * -1, otherwise performs an `n`-point inverse FFT. \returns the FFT result.
 */
template <typename DeviceVec, typename Type>
inline decltype(auto) irfft(dense_vector<DeviceVec, complex<Type>> const& vec,
                            int n = -1);

/**
 *  Performs complex-to-complex FFT.
 *  \param vec the vector to perform the fft on.
 *  \param direction fft_direction::forward for forward fft,
 * fft_direction::inverse for inverse normalized fft. \returns the FFT result.
 */
template <typename DeviceVec, typename Type>
inline decltype(auto) cfft(dense_vector<DeviceVec, complex<Type>> const& vec,
                           fft_direction direction);

/**
 *  Performs 2D real-to-complex forward FFT.
 *  \param mat the matrix to perform the fft on.
 *  \returns the FFT result.
 */
template <typename DeviceMat, typename Type>
inline decltype(auto) rfft(dense_matrix<DeviceMat, Type> const& mat);

/**
 * Performs 2D complex-to-real inverse normalized FFT.
 * \param mat the matrix to perform the fft on.
 * \param n the number of rows. if n == -1, the number is deduced from the size
 * of `mat`. \returns the FFT result.
 */
template <typename DeviceMat, typename Type>
inline decltype(auto) irfft(dense_matrix<DeviceMat, complex<Type>> const& mat,
                            int n = -1);

/**
 *  Performs 2D complex-to-complex FFT.
 *  \param mat the matrix to perform the fft on.
 *  \param direction fft_direction::forward for forward fft,
 * fft_direction::inverse for inverse normalized fft. \returns the FFT result.
 */
template <typename DeviceMat, typename Type>
inline decltype(auto) cfft(dense_matrix<DeviceMat, complex<Type>> const& mat,
                           fft_direction direction);
}  // namespace strict
}  // namespace mgcpp

#include <mgcpp/operations/fft.tpp>

#endif
