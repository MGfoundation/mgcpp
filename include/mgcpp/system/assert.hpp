
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_SYSTEM_ASSERT_HPP_
#define _MGCPP_SYSTEM_ASSERT_HPP_

namespace mgcpp
{
    inline bool
    ASSERT_MESSAGE(char const* message)
    {
        (void)message;
        return false;
    }
}

#ifndef MGCPP_ASSERT
#include <cassert>
#define MGCPP_ASSERT(EXPR, MESSAGE)                 \
    assert(( EXPR ) || mgcpp::ASSERT_MESSAGE(MESSAGE)) 
#endif

#endif
