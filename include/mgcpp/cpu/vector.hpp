
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CPU_VECTOR_HPP_
#define _MGCPP_CPU_VECTOR_HPP_

#include <cstdlib>

#include <mgcpp/cpu/forward.hpp>
#include <mgcpp/global/allignment.hpp>

namespace mgcpp
{
    namespace cpu 
    {
        template<typename T,
                 allignment Allign>
        class vector
        {
        private:
            T* _data; 
            size_t _size;
        
        public:
            inline vector() noexcept;

            inline ~vector() noexcept;

            inline vector(size_t size);

            inline vector(size_t size, T init);

            inline vector(size_t size, T* data) noexcept;

            // template<size_t DeviceId>
            // inline vector(
            //     gpu::vector<T, DeviceId> const& gpu_mat);

            // template<size_t DeviceId>
            // inline gpu::vector<T, DeviceId, >
            // copy_to_gpu() const;

            inline T
            operator[](size_t i) const;

            inline T&
            operator[](size_t i);

            inline T const*
            get_data() const;

            inline T*
            get_data_mutable() const;

            inline size_t
            shape() const noexcept;

            inline size_t
            size() const noexcept;
        };
    }
}

#include <mgcpp/cpu/vector.tpp>
#endif
