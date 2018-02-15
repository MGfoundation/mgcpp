
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Example code for executing Multi-layer perception model
// for the MNIST dataset.

#include <mgcpp/expressions/dmat_dvec_mult.hpp>
#include <mgcpp/expressions/dvec_dvec_add.hpp>
#include <mgcpp/expressions/dvec_elemwise.hpp>
#include <mgcpp/mgcpp.hpp>
#include <mgcpp/operations/trans.hpp>

#include "model.h"

template <size_t NIn,
          size_t NOut,
          float (&Weights)[NIn][NOut],
          float (&Bias)[NOut]>
struct Layer {
  using matrix = mgcpp::device_matrix<float>;
  using vector = mgcpp::device_vector<float>;
  matrix W;
  vector b;

  Layer() {
    W = mgcpp::strict::trans(matrix::from_c_array(Weights));
    b = vector(NOut, Bias);
  }

  template <typename T>
  auto operator()(T const& input) {
    return mgcpp::relu(W * input + b);
  }
};

int main() {
  mgcpp::device_vector<float> input(28 * 28);
  Layer<28 * 28, 256, w1, b1> l_input;
  auto y1 = l_input(input);
  Layer<256, 256, w2, b2> l_hidden;
  auto y2 = l_hidden(y1);
  Layer<256, 10, w3, b3> l_output;
  auto result = l_output(y2);
  auto ans = result.eval();
  for (int i = 0; i < 10; ++i)
    std::cout << i << ": " << ans.check_value(i) << std::endl;
}
