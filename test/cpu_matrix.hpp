#ifndef _CPU_MATRIX_HPP_
#define _CPU_MATRIX_HPP_

#include <cstdlib>
#include <mgcpp/matrix/device_matrix.hpp>
#include <vector>

inline size_t encode_index(size_t i, size_t j, size_t m) {
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

template <typename T>
class cpu_vector {
 private:
  size_t _m;
  T* _data;

 public:
  cpu_vector(size_t m) : _m(m) { _data = (T*)malloc(sizeof(T) * _m); }

  T& operator()(size_t i) { return _data[i]; }

  T* data() const { return _data; }

  size_t size() const { return _m; }

  mgcpp::shape<1> shape() const { return {_m}; }

  ~cpu_vector() { free(_data); }
};

namespace mgcpp {
template <typename Type>
struct adapter<cpu_matrix<Type>> : std::true_type {
  void operator()(cpu_matrix<Type> const& mat,
                  Type** out_p,
                  size_t* m,
                  size_t* n) {
    *out_p = mat.data();
    auto shape = mat.shape();
    *m = shape[0];
    *n = shape[1];
  }
};

template <typename Type>
struct adapter<cpu_vector<Type>> : std::true_type {
  void operator()(cpu_vector<Type> const& vec, Type** out_p, size_t* m) {
    *out_p = vec.data();
    *m = vec.size();
  }
};

// template <typename Type>
// struct adapter<std::vector<Type>> : std::true_type {
//     void operator()(std::vector<Type> const& vec, Type** out_p, size_t* m) {
//         *out_p = vec.data();
//         *m = vec.size();
//     }
// };

}  // namespace mgcpp

#endif
