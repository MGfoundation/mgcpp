
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_ALLOCATOR_HPP_
#define _MGCPP_TYPE_TRAITS_ALLOCATOR_HPP_

#include <cstdlib>

namespace mgcpp
{
    template<typename... Type>
    struct change_allocator_type {};

    template<template<typename, size_t> class Allocator,
             size_t DeviceId,
             typename OldType,
             typename NewType>
    struct change_allocator_type<Allocator<OldType, DeviceId>,
                                 NewType>
    {
        using type = Allocator<NewType, DeviceId>;
    };
}

#endif
