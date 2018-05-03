
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef EVAL_CONTEXT_HPP
#define EVAL_CONTEXT_HPP

namespace mgcpp {
struct eval_context {
    int total_computations = 0;
    int cache_hits = 0;
};
}

#endif // EVAL_CONTEXT_HPP
