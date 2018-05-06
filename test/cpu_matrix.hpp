#ifndef _CPU_MATRIX_HPP_
#define _CPU_MATRIX_HPP_

#include <cstdlib>
#include <mgcpp/matrix/device_matrix.hpp>

size_t encode_index(size_t i, size_t j, size_t m) {
    return i + j * m;
}

template <typename T>
class cpu_matrix {
private:
    size_t _m;
    size_t _n;
    T* _data;

public:
    cpu_matrix(size_t m, size_t n) : _m(m), _n(n) {
        _data = (T*)malloc(sizeof(T) * _m * _n);
    }

    T& operator()(size_t i, size_t j) { return _data[encode_index(i, j, _m)]; }

    T* data() const { return _data; }

    mgcpp::shape<2> shape() const { return {_m, _n}; }

    ~cpu_matrix() { free(_data); }
};


namespace mgcpp {
    template <typename T>
    struct adapter<cpu_matrix<T>> : std::true_type {
        void operator()(cpu_matrix<T> const& mat, T** out_p, size_t* m, size_t* n) {
            *out_p = mat.data();
            auto shape = mat.shape();
            *m = shape[0];
            *n = shape[1];
        }
    };
}  // namespace mgcpp


#endif
