
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Example code for executing Multi-layer perception model
// for the MNIST dataset.

#include <mgcpp/expressions/dmat_dvec_mult.hpp>
#include <mgcpp/expressions/dvec_dvec_add.hpp>
#include <mgcpp/expressions/dvec_elemwise.hpp>
#include <mgcpp/operations/trans.hpp>
#include <mgcpp/mgcpp.hpp>

//#include "model.h"

using vector = mgcpp::device_vector<float>;
vector b;

auto f ()
{
    return b + b + b + b + b + b + b;
}

int main() {
  b = vector(256);
  auto y1 = f();

  auto ans = mgcpp::eval(y1);
}
