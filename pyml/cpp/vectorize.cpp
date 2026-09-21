#include "vectorize.hpp"
#include <cstddef>
#include <vector>
#include <immintrin.h>

namespace VecMnpUtils {

#if defined(__SSE2__)
struct SseOps {
    using VECOBJ = __m128;
    static constexpr int WIDTH = 4;

    static VECOBJ LOADVPS(const float* p) noexcept { return _mm_loadu_ps(p); }
    static void STOREVPS(float* p, VECOBJ v) noexcept { _mm_storeu_ps(p, v); }
    static VECOBJ MULTVPS(VECOBJ a, VECOBJ b) noexcept { return _mm_mul_ps(a, b); }
    static VECOBJ ADDVPS(VECOBJ a, VECOBJ b) noexcept { return _mm_add_ps(a, b); }
    static VECOBJ SETVPS(float v) noexcept { return _mm_set1_ps(v); }

    using VECOBJ_PD = __m128d;
    static constexpr int WIDTH_PD = 2;
    static VECOBJ_PD LOADVPD(const double* p) noexcept { return _mm_loadu_pd(p); }
    static void STOREVPD(double* p, VECOBJ_PD v) noexcept { _mm_storeu_pd(p, v); }
    static VECOBJ_PD MULTVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return _mm_mul_pd(a, b); }
    static VECOBJ_PD ADDVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return _mm_add_pd(a, b); }
    static VECOBJ_PD SETVPD(double v) noexcept { return _mm_set1_pd(v); }
};
#endif

#if defined(__AVX__)
struct AvxOps {
    using VECOBJ = __m256;
    static constexpr int WIDTH = 8;

    static VECOBJ LOADVPS(const float* p) noexcept { return _mm256_loadu_ps(p); }
    static void STOREVPS(float* p, VECOBJ v) noexcept { _mm256_storeu_ps(p, v); }
    static VECOBJ MULTVPS(VECOBJ a, VECOBJ b) noexcept { return _mm256_mul_ps(a, b); }
    static VECOBJ ADDVPS(VECOBJ a, VECOBJ b) noexcept { return _mm256_add_ps(a, b); }
    static VECOBJ SETVPS(float v) noexcept { return _mm256_set1_ps(v); }

    using VECOBJ_PD = __m256d;
    static constexpr int WIDTH_PD = 4;
    static VECOBJ_PD LOADVPD(const double* p) noexcept { return _mm256_loadu_pd(p); }
    static void STOREVPD(double* p, VECOBJ_PD v) noexcept { _mm256_storeu_pd(p, v); }
    static VECOBJ_PD MULTVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return _mm256_mul_pd(a, b); }
    static VECOBJ_PD ADDVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return _mm256_add_pd(a, b); }
    static VECOBJ_PD SETVPD(double v) noexcept { return _mm256_set1_pd(v); }
};
#endif

#if defined(__AVX512F__)
struct Avx512Ops {
    using VECOBJ = __m512;
    static constexpr int WIDTH = 16;

    static VECOBJ LOADVPS(const float* p) noexcept { return _mm512_loadu_ps(p); }
    static void STOREVPS(float* p, VECOBJ v) noexcept { _mm512_storeu_ps(p, v); }
    static VECOBJ MULTVPS(VECOBJ a, VECOBJ b) noexcept { return _mm512_mul_ps(a, b); }
    static VECOBJ ADDVPS(VECOBJ a, VECOBJ b) noexcept { return _mm512_add_ps(a, b); }
    static VECOBJ SETVPS(float v) noexcept { return _mm512_set1_ps(v); }

    using VECOBJ_PD = __m512d;
    static constexpr int WIDTH_PD = 8;
    static VECOBJ_PD LOADVPD(const double* p) noexcept { return _mm512_loadu_pd(p); }
    static void STOREVPD(double* p, VECOBJ_PD v) noexcept { _mm512_storeu_pd(p, v); }
    static VECOBJ_PD MULTVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return _mm512_mul_pd(a, b); }
    static VECOBJ_PD ADDVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return _mm512_add_pd(a, b); }
    static VECOBJ_PD SETVPD(double v) noexcept { return _mm512_set1_pd(v); }
};
#endif

struct ScalarOps {
    using VECOBJ = float;
    static constexpr int WIDTH = 1;

    static VECOBJ LOADVPS(const float* p) noexcept { return *p; }
    static void STOREVPS(float* p, VECOBJ v) noexcept { *p = v; }
    static VECOBJ MULTVPS(VECOBJ a, VECOBJ b) noexcept { return a * b; }
    static VECOBJ ADDVPS(VECOBJ a, VECOBJ b) noexcept { return a + b; }
    static VECOBJ SETVPS(float v) noexcept { return v; }

    using VECOBJ_PD = double;
    static constexpr int WIDTH_PD = 1;
    static VECOBJ_PD LOADVPD(const double* p) noexcept { return *p; }
    static void STOREVPD(double* p, VECOBJ_PD v) noexcept { *p = v; }
    static VECOBJ_PD MULTVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return a * b; }
    static VECOBJ_PD ADDVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return a + b; }
    static VECOBJ_PD SETVPD(double v) noexcept { return v; }
};

#if defined(__AVX512F__)
using DefaultOps = Avx512Ops;
#elif defined(__AVX__)
using DefaultOps = AvxOps;
#elif defined(__SSE2__)
using DefaultOps = SseOps;
#else
using DefaultOps = ScalarOps;
#endif

template <typename Ops>
std::vector<float> SinglePrecisionMultiplier(const float* queue1,
                                             const float* queue2,
                                             int qlength) {
    
    if (qlength <= 0) return {};
    std::vector<float> res(qlength);
    int offset = 0;

    for (; offset + Ops::WIDTH <= qlength; offset += Ops::WIDTH) {
        auto vecq1 = Ops::LOADVPS(queue1 + offset);
        auto vecq2 = Ops::LOADVPS(queue2 + offset);
        auto vecr  = Ops::MULTVPS(vecq1, vecq2);
        Ops::STOREVPS(res.data() + offset, vecr);
    }

    for (; offset < qlength; ++offset) {
        res[offset] = queue1[offset] * queue2[offset];
    }

    return res;
}

template <typename Ops>
std::vector<double> DoublePrecisionMultiplier(const double* queue1,
                                              const double* queue2,
                                              int qlength) {
    if (qlength <= 0) return {};
    std::vector<double> res(qlength);
    int offset = 0;

    for (; offset + Ops::WIDTH_PD <= qlength; offset += Ops::WIDTH_PD) {
        auto vecq1 = Ops::LOADVPD(queue1 + offset);
        auto vecq2 = Ops::LOADVPD(queue2 + offset);
        auto vecr  = Ops::MULTVPD(vecq1, vecq2);
        Ops::STOREVPD(res.data() + offset, vecr);
    }

    for (; offset < qlength; ++offset) {
        res[offset] = queue1[offset] * queue2[offset];
    }

    return res;
}

template <Vectorable PRECISION> std::vector<PRECISION> Multiplier(const PRECISION* queue1,
                                                                  const PRECISION* queue2,
                                                                  int qlength) {
    if constexpr (std::is_same_v<PRECISION, float>)
        return SinglePrecisionMultiplier<DefaultOps>(queue1, queue2, qlength);
    if constexpr (std::is_same_v<PRECISION, double>)
        return DoublePrecisionMultiplier<DefaultOps>(queue1, queue2, qlength);
}

template <typename Ops>
void MultiplyAccumulateFloat(float* destination, const float* values,
                             float coefficient, int length) {
    int offset = 0;
    for (; offset + Ops::WIDTH <= length; offset += Ops::WIDTH) {
        const auto value = Ops::LOADVPS(values + offset);
        const auto scale = Ops::SETVPS(coefficient);
        Ops::STOREVPS(destination + offset,
                      Ops::ADDVPS(Ops::LOADVPS(destination + offset),
                                  Ops::MULTVPS(value, scale)));
    }
    for (; offset < length; ++offset)
        destination[offset] += coefficient * values[offset];
}

template <typename Ops>
void MultiplyAccumulateDouble(double* destination, const double* values,
                              double coefficient, int length) {
    int offset = 0;
    for (; offset + Ops::WIDTH_PD <= length; offset += Ops::WIDTH_PD) {
        const auto value = Ops::LOADVPD(values + offset);
        const auto scale = Ops::SETVPD(coefficient);
        Ops::STOREVPD(destination + offset,
                      Ops::ADDVPD(Ops::LOADVPD(destination + offset),
                                  Ops::MULTVPD(value, scale)));
    }
    for (; offset < length; ++offset)
        destination[offset] += coefficient * values[offset];
}

template <Vectorable PRECISION>
void MultiplyAccumulate(PRECISION* destination, const PRECISION* values,
                        PRECISION coefficient, int length) {
    if (length <= 0) return;
    if constexpr (std::is_same_v<PRECISION, float>)
        MultiplyAccumulateFloat<DefaultOps>(destination, values, coefficient, length);
    else
        MultiplyAccumulateDouble<DefaultOps>(destination, values, coefficient, length);
}

template std::vector<float> Multiplier<float>(const float*, const float*, int);
template std::vector<double> Multiplier<double>(const double*, const double*, int);
template void MultiplyAccumulate<float>(float*, const float*, float, int);
template void MultiplyAccumulate<double>(double*, const double*, double, int);
}
