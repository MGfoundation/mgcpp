
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_SYSTEM_CONCEPT_HPP_
#define _MGCPP_SYSTEM_CONCEPT_HPP_

#define MGCPP_CONCEPT(...)                                      \
    typename = typename std::enable_if<(__VA_ARGS__)>::type 

#endif
