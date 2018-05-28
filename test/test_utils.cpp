
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <boost/random/mersenne_twister.hpp>

// https://github.com/s9w/articles/blob/master/perf%20cpp%20random.md
// according to this benchmark, boost mt is the fastest
// but anything fast will just be fine
boost::random::mt19937 rng(MGCPP_RAND_SEED);
std::uniform_real_distribution<double> dist(0.0, 1.0);

namespace internal {
double uniform_rand() {
  return dist(rng);
}
}  // namespace internal
