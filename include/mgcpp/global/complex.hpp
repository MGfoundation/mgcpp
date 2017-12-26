
#ifndef _MGCPP_GLOBAL_COMPLEX_HPP_
#define _MGCPP_GLOBAL_COMPLEX_HPP_

#include <cmath>
#include <ostream>

namespace mgcpp
{
    template<typename T>
    struct complex {
        T real, imag;

        complex() : real{}, imag{} {}
        complex(T x) : real{x}, imag{} {}
        complex(T real, T imag) : real{real}, imag{imag} {}

        complex operator* (T rhs) const {
            return { real * rhs, imag * rhs };
        }

        complex& operator* (T rhs) {
            return *this = *this * rhs;
        }

        complex operator* (complex rhs) const {
            return {
                real * rhs.real - imag * rhs.imag,
                real * rhs.imag + imag * rhs.real
            };
        }

        complex& operator*= (complex rhs) {
            return *this = *this * rhs;
        }

        complex operator/ (T rhs) const {
            return { real / rhs, imag / rhs };
        }

        complex& operator/= (T rhs) {
            return *this = *this / rhs;
        }

        complex operator+ (complex rhs) const {
            return {
                real + rhs.real,
                imag + rhs.imag
            };
        }

        complex& operator+= (complex rhs) {
            return *this = *this + rhs;
        }

        complex operator- (complex rhs) const {
            return {
                real - rhs.real,
                imag - rhs.imag
            };
        }

        complex& operator-= (complex rhs) {
            return *this = *this - rhs;
        }
    };

    template<typename T>
    std::ostream& operator<< (std::ostream& os, complex<T> x) {
        std::ios::fmtflags f(os.flags());
        os << x.real << std::showpos << x.imag << 'i';
        os.flags(f);
        return os;
    }

    template<typename T>
    mgcpp::complex<T> polar(T r, T theta) {
        return {r * std::cos(theta), r * std::sin(theta)};
    }
}

namespace std
{
    template<typename T>
    T abs(mgcpp::complex<T> x) {
        return x.real * x.real + x.imag * x.imag;
    }
}

#endif
