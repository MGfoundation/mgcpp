
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Example code for executing Multi-layer perception model
// for the MNIST dataset.

#include <mgcpp/expressions/dmat_dvec_mult.hpp>
#include <mgcpp/expressions/dvec_dvec_add.hpp>
#include <mgcpp/mgcpp.hpp>

struct HiddenLayer {
  mgcpp::device_matrix<float> W;
  mgcpp::device_vector<float> b;

  HiddenLayer(int id)
    : W(mgcpp::make_shape(28 * 28, 500)),
      b(500)
  {
    // TODO: retrieve model from a file
  }

  template <typename T>
  auto operator()(T const& input) {
    return mgcpp::relu(W * input + b);
  }
};

int main() {
  mgcpp::device_vector<float> input(28 * 28);
  HiddenLayer hidden_0(0);
  auto y1 = hidden_0(input);
  HiddenLayer hidden_1(1);
  auto y2 = hidden_1(y1);
  auto result = mgcpp::softmax(y2);
  auto ans = result.eval();
}
