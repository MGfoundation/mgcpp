
#include <mgcpp/system/type_cast.hpp>

#include <vector_types.h>
#include <cuda_fp16.h>

#include <immintrin.h>
#include <cassert>
#include <cstring>

#ifdef __GNUC__
#define F16C_TARGET __attribute__((target("f16c")))
#else
#define F16C_TARGET
#endif

namespace mgcpp
{
    F16C_TARGET
    void half_to_float_impl(__half const* first, __half const* last, float* d_first)
    {
        size_t size = last - first;
        for (auto i = 0u; i < size; ++i) {
            uint16_t s = reinterpret_cast<const uint16_t&>(first[i]);
            __m128 mm1 = _mm_cvtph_ps(_mm_set1_epi16(s));
            std::memcpy(d_first + i, &mm1, 4);
        }
    }

    F16C_TARGET
    void float_to_half_impl(float const* first, float const* last, __half* d_first)
    {
        size_t size = last - first;
        for (auto i = 0u; i < size; ++i) {
            __m128i mm1 = _mm_cvtps_ph(_mm_set1_ps(first[i]), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
            std::memcpy(d_first + i, &mm1, 2);
        }
    }
}
