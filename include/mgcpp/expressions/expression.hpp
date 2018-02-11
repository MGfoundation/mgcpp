
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_EXPRESSION_HPP_
#define _MGCPP_EXPRESSIONS_EXPRESSION_HPP_

namespace mgcpp
{
    template<typename Type>
    struct expression
    {
        inline Type&
        operator~() noexcept
        { return *static_cast<Type*>(this); }

        inline Type const&
        operator~() const noexcept
        { return *static_cast<Type const*>(this); }
    };
}

#endif
