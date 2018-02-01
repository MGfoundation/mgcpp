
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TEST_TEST_POLICY_HPP_
#define _MGCPP_TEST_TEST_POLICY_HPP_

namespace mgcpp
{
    class test_policy
    {
        test_policy();
        
        size_t _device_num;
        bool _detect_memory_leak;

    public:
        static test_policy&
        get_policy();

        size_t
        device_num() const noexcept;

        bool
        detect_memory_leak() const noexcept;
    };
}

#endif
