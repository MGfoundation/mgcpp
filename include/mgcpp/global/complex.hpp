
#ifndef _MGCPP_GLOBAL_COMPLEX_HPP_
#define _MGCPP_GLOBAL_COMPLEX_HPP_

#include <cmath>
#include <ostream>

#ifdef __CUDACC__
#define HOST_AND_DEVICE __host__ __device__
#else
#include <type_traits>
#define HOST_AND_DEVICE
#endif

namespace mgcpp
{
    template<typename T>
    struct complex {
        T real, imag;

        // constructors are problematic in nvcc
#ifndef __CUDACC__
        complex() : real{}, imag{} {}
        explicit complex(T x) : real{x}, imag{} {}
        complex(T real, T imag) : real{real}, imag{imag} {}
#endif

        HOST_AND_DEVICE
        complex operator* (T rhs) const {
            complex r = { real * rhs, imag * rhs };
            return r;
        }

        HOST_AND_DEVICE
        complex& operator* (T rhs) {
            return *this = *this * rhs;
        }

        HOST_AND_DEVICE
        complex operator* (complex rhs) const {
            complex r = {
                real * rhs.real - imag * rhs.imag,
                real * rhs.imag + imag * rhs.real
            };
            return r;
        }

        HOST_AND_DEVICE
        complex& operator*= (complex rhs) {
            return *this = *this * rhs;
        }

        HOST_AND_DEVICE
        complex operator/ (T rhs) const {
            complex r = { real / rhs, imag / rhs };
            return r;
        }

        HOST_AND_DEVICE
        complex& operator/= (T rhs) {
            return *this = *this / rhs;
        }

        HOST_AND_DEVICE
        complex operator+ (complex rhs) const {
            complex r = {
                real + rhs.real,
                imag + rhs.imag
            };
            return r;
        }

        HOST_AND_DEVICE
        complex& operator+= (complex rhs) {
            return *this = *this + rhs;
        }

        HOST_AND_DEVICE
        complex operator- (complex rhs) const {
            complex r = {
                real - rhs.real,
                imag - rhs.imag
            };
            return r;
        }

        HOST_AND_DEVICE
        complex& operator-= (complex rhs) {
            return *this = *this - rhs;
        }
    };

#ifndef __CUDACC__
    static_assert(std::is_standard_layout<complex<float>>::value, "mgcpp::complex<float> is not standard layout");
    static_assert(std::is_standard_layout<complex<double>>::value, "mgcpp::complex<double> is not standard layout");
#endif

    template<typename T>
    std::ostream& operator<< (std::ostream& os, complex<T> x) {
        std::ios::fmtflags f(os.flags());
        os << x.real << std::showpos << x.imag << 'i';
        os.flags(f);
        return os;
    }

    template<typename T>
    HOST_AND_DEVICE
    mgcpp::complex<T> polar(T r, T theta) {
        mgcpp::complex<T> x = {r * std::cos(theta), r * std::sin(theta)};
        return x;
    }
}

namespace std
{
    template<typename T>
    HOST_AND_DEVICE
    T abs(mgcpp::complex<T> x) {
        return x.real * x.real + x.imag * x.imag;
    }
}

#endif
