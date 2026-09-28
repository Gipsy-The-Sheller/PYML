#include "vectorize.hpp"
#include <cstddef>
#include <cstring>
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

    // a*b + c.  Fused when the target has FMA, so the accumulate stays one
    // rounded operation instead of two.  Falls back to mul-then-add.
#if defined(__FMA__)
    static VECOBJ FMADDVPS(VECOBJ a, VECOBJ b, VECOBJ c) noexcept { return _mm_fmadd_ps(a, b, c); }
#else
    static VECOBJ FMADDVPS(VECOBJ a, VECOBJ b, VECOBJ c) noexcept { return _mm_add_ps(_mm_mul_ps(a, b), c); }
#endif

    using VECOBJ_PD = __m128d;
    static constexpr int WIDTH_PD = 2;
    static VECOBJ_PD LOADVPD(const double* p) noexcept { return _mm_loadu_pd(p); }
    static void STOREVPD(double* p, VECOBJ_PD v) noexcept { _mm_storeu_pd(p, v); }
    static VECOBJ_PD MULTVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return _mm_mul_pd(a, b); }
    static VECOBJ_PD ADDVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return _mm_add_pd(a, b); }
    static VECOBJ_PD SETVPD(double v) noexcept { return _mm_set1_pd(v); }

#if defined(__FMA__)
    static VECOBJ_PD FMADDVPD(VECOBJ_PD a, VECOBJ_PD b, VECOBJ_PD c) noexcept { return _mm_fmadd_pd(a, b, c); }
#else
    static VECOBJ_PD FMADDVPD(VECOBJ_PD a, VECOBJ_PD b, VECOBJ_PD c) noexcept { return _mm_add_pd(_mm_mul_pd(a, b), c); }
#endif
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

#if defined(__FMA__)
    static VECOBJ FMADDVPS(VECOBJ a, VECOBJ b, VECOBJ c) noexcept { return _mm256_fmadd_ps(a, b, c); }
#else
    static VECOBJ FMADDVPS(VECOBJ a, VECOBJ b, VECOBJ c) noexcept { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
#endif

    using VECOBJ_PD = __m256d;
    static constexpr int WIDTH_PD = 4;
    static VECOBJ_PD LOADVPD(const double* p) noexcept { return _mm256_loadu_pd(p); }
    static void STOREVPD(double* p, VECOBJ_PD v) noexcept { _mm256_storeu_pd(p, v); }
    static VECOBJ_PD MULTVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return _mm256_mul_pd(a, b); }
    static VECOBJ_PD ADDVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return _mm256_add_pd(a, b); }
    static VECOBJ_PD SETVPD(double v) noexcept { return _mm256_set1_pd(v); }

#if defined(__FMA__)
    static VECOBJ_PD FMADDVPD(VECOBJ_PD a, VECOBJ_PD b, VECOBJ_PD c) noexcept { return _mm256_fmadd_pd(a, b, c); }
#else
    static VECOBJ_PD FMADDVPD(VECOBJ_PD a, VECOBJ_PD b, VECOBJ_PD c) noexcept { return _mm256_add_pd(_mm256_mul_pd(a, b), c); }
#endif
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
    // fused multiply-add is part of AVX512F itself
    static VECOBJ FMADDVPS(VECOBJ a, VECOBJ b, VECOBJ c) noexcept { return _mm512_fmadd_ps(a, b, c); }

    using VECOBJ_PD = __m512d;
    static constexpr int WIDTH_PD = 8;
    static VECOBJ_PD LOADVPD(const double* p) noexcept { return _mm512_loadu_pd(p); }
    static void STOREVPD(double* p, VECOBJ_PD v) noexcept { _mm512_storeu_pd(p, v); }
    static VECOBJ_PD MULTVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return _mm512_mul_pd(a, b); }
    static VECOBJ_PD ADDVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return _mm512_add_pd(a, b); }
    static VECOBJ_PD SETVPD(double v) noexcept { return _mm512_set1_pd(v); }
    static VECOBJ_PD FMADDVPD(VECOBJ_PD a, VECOBJ_PD b, VECOBJ_PD c) noexcept { return _mm512_fmadd_pd(a, b, c); }
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
    static VECOBJ FMADDVPS(VECOBJ a, VECOBJ b, VECOBJ c) noexcept { return a * b + c; }

    using VECOBJ_PD = double;
    static constexpr int WIDTH_PD = 1;
    static VECOBJ_PD LOADVPD(const double* p) noexcept { return *p; }
    static void STOREVPD(double* p, VECOBJ_PD v) noexcept { *p = v; }
    static VECOBJ_PD MULTVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return a * b; }
    static VECOBJ_PD ADDVPD(VECOBJ_PD a, VECOBJ_PD b) noexcept { return a + b; }
    static VECOBJ_PD SETVPD(double v) noexcept { return v; }
    static VECOBJ_PD FMADDVPD(VECOBJ_PD a, VECOBJ_PD b, VECOBJ_PD c) noexcept { return a * b + c; }
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
    const auto scale = Ops::SETVPS(coefficient);
    for (; offset + Ops::WIDTH <= length; offset += Ops::WIDTH) {
        const auto value = Ops::LOADVPS(values + offset);
        Ops::STOREVPS(destination + offset,
                      Ops::FMADDVPS(value, scale, Ops::LOADVPS(destination + offset)));
    }
    for (; offset < length; ++offset)
        destination[offset] += coefficient * values[offset];
}

template <typename Ops>
void MultiplyAccumulateDouble(double* destination, const double* values,
                              double coefficient, int length) {
    int offset = 0;
    const auto scale = Ops::SETVPD(coefficient);
    for (; offset + Ops::WIDTH_PD <= length; offset += Ops::WIDTH_PD) {
        const auto value = Ops::LOADVPD(values + offset);
        Ops::STOREVPD(destination + offset,
                      Ops::FMADDVPD(value, scale, Ops::LOADVPD(destination + offset)));
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

template <typename Ops>
void MultiplyInPlaceFloat(float* destination, const float* values, int length) {
    int offset = 0;
    for (; offset + Ops::WIDTH <= length; offset += Ops::WIDTH) {
        Ops::STOREVPS(destination + offset,
                      Ops::MULTVPS(Ops::LOADVPS(destination + offset),
                                  Ops::LOADVPS(values + offset)));
    }
    for (; offset < length; ++offset)
        destination[offset] *= values[offset];
}

template <typename Ops>
void MultiplyInPlaceDouble(double* destination, const double* values, int length) {
    int offset = 0;
    for (; offset + Ops::WIDTH_PD <= length; offset += Ops::WIDTH_PD) {
        Ops::STOREVPD(destination + offset,
                      Ops::MULTVPD(Ops::LOADVPD(destination + offset),
                                   Ops::LOADVPD(values + offset)));
    }
    for (; offset < length; ++offset)
        destination[offset] *= values[offset];
}

template <Vectorable PRECISION>
void MultiplyInPlace(PRECISION* destination, const PRECISION* values, int length) {
    if (length <= 0) return;
    if constexpr (std::is_same_v<PRECISION, float>)
        MultiplyInPlaceFloat<DefaultOps>(destination, values, length);
    else
        MultiplyInPlaceDouble<DefaultOps>(destination, values, length);
}

template <typename Ops>
void ScaleInPlaceFloat(float* destination, float coefficient, int length) {
    int offset = 0;
    const auto scale = Ops::SETVPS(coefficient);
    for (; offset + Ops::WIDTH <= length; offset += Ops::WIDTH)
        Ops::STOREVPS(destination + offset,
                      Ops::MULTVPS(Ops::LOADVPS(destination + offset), scale));
    for (; offset < length; ++offset)
        destination[offset] *= coefficient;
}

template <typename Ops>
void ScaleInPlaceDouble(double* destination, double coefficient, int length) {
    int offset = 0;
    const auto scale = Ops::SETVPD(coefficient);
    for (; offset + Ops::WIDTH_PD <= length; offset += Ops::WIDTH_PD)
        Ops::STOREVPD(destination + offset,
                      Ops::MULTVPD(Ops::LOADVPD(destination + offset), scale));
    for (; offset < length; ++offset)
        destination[offset] *= coefficient;
}

template <Vectorable PRECISION>
void ScaleInPlace(PRECISION* destination, PRECISION coefficient, int length) {
    if (length <= 0) return;
    if constexpr (std::is_same_v<PRECISION, float>)
        ScaleInPlaceFloat<DefaultOps>(destination, coefficient, length);
    else
        ScaleInPlaceDouble<DefaultOps>(destination, coefficient, length);
}

/* ------------------------------------------------------- PmatMul --
   parent[j*npatt+h] = sum_k P[j*nstates+k] * child[k*npatt+h].

   The pattern tile is loaded once, the k-reduction happens in registers, and
   the parent tile is stored once.  Doing this as nstates*nstates AXPYs
   instead would re-stream both blocks for every (j,k) pair -- nstates times
   the traffic for the same arithmetic, which for the 4-state case loses more
   to the memory system than SIMD wins back in instructions.

   NS must be a compile-time constant.  With a runtime `nstates` the compiler
   has to keep the register tiles cv[]/pv[] on the stack, because a runtime
   index cannot be proven to stay in range -- and every fused multiply then
   reads its operand back out of the stack, which measured 1.5x SLOWER than
   the plain scalar loop it was meant to replace.  Specialising the state
   count is what makes the kernel actually pay.                              */

template <int NS, bool ACCUMULATE, typename Ops>
void PmatMulFixedFloat(float* parent, const float* child, const float* P,
                       int npatt) {
    using V = typename Ops::VECOBJ;
    constexpr int W = Ops::WIDTH;
    V cv[NS], pv[NS];
    int h = 0, j, k;

    for (; h + W <= npatt; h += W) {
        for (k = 0; k < NS; ++k)
            cv[k] = Ops::LOADVPS(child + (std::size_t)k * npatt + h);
        for (j = 0; j < NS; ++j) {
            for (k = 0; k < NS; ++k)
                pv[k] = Ops::SETVPS(P[j * NS + k]);
            V acc = Ops::MULTVPS(cv[0], pv[0]);
            for (k = 1; k < NS; ++k)
                acc = Ops::FMADDVPS(cv[k], pv[k], acc);
            float* out = parent + (std::size_t)j * npatt + h;
            if (ACCUMULATE) acc = Ops::MULTVPS(Ops::LOADVPS(out), acc);
            Ops::STOREVPS(out, acc);
        }
    }
    if (h < npatt) {                       /* see the double version */
        alignas(64) float cin[NS][W], cout[NS][W];
        const int rem = npatt - h;
        for (k = 0; k < NS; ++k) {
            for (j = 0; j < W; ++j) cin[k][j] = 0.0f;
            std::memcpy(cin[k], child + (std::size_t)k * npatt + h,
                        (std::size_t)rem * sizeof(float));
        }
        for (j = 0; j < NS; ++j) {
            for (k = 0; k < W; ++k) cout[j][k] = 0.0f;
            if (ACCUMULATE)
                std::memcpy(cout[j], parent + (std::size_t)j * npatt + h,
                            (std::size_t)rem * sizeof(float));
        }

        for (k = 0; k < NS; ++k) cv[k] = Ops::LOADVPS(cin[k]);
        for (j = 0; j < NS; ++j) {
            for (k = 0; k < NS; ++k)
                pv[k] = Ops::SETVPS(P[j * NS + k]);
            V acc = Ops::MULTVPS(cv[0], pv[0]);
            for (k = 1; k < NS; ++k)
                acc = Ops::FMADDVPS(cv[k], pv[k], acc);
            if (ACCUMULATE) acc = Ops::MULTVPS(Ops::LOADVPS(cout[j]), acc);
            Ops::STOREVPS(cout[j], acc);
        }
        for (j = 0; j < NS; ++j)
            std::memcpy(parent + (std::size_t)j * npatt + h, cout[j],
                        (std::size_t)rem * sizeof(float));
    }
}

/* ACCUMULATE = false:  parent  = sum_k P[j][k] * child[k][h]
   ACCUMULATE = true :  parent *= sum_k P[j][k] * child[k][h]
   The second form is what a node with more than one child needs, and folding
   the multiply into the same pass avoids a whole extra sweep over the parent
   block -- which for a two-child node is half the work.

   The ragged end of the pattern axis is handled by padding: the last npatt%W
   patterns are copied into a zeroed tile, run through the same vector code,
   and copied back.  Padding a sum with zeros is harmless, and doing it this
   way means there is only ONE implementation of the arithmetic, so the tail
   rounds identically to the body.  The earlier scalar tail was a second
   implementation that had to be kept numerically in step by hand.            */
template <int NS, bool ACCUMULATE, typename Ops>
void PmatMulFixedDouble(double* parent, const double* child, const double* P,
                        int npatt) {
    using V = typename Ops::VECOBJ_PD;
    constexpr int W = Ops::WIDTH_PD;
    V cv[NS], pv[NS];
    int h = 0, j, k;

    for (; h + W <= npatt; h += W) {
        for (k = 0; k < NS; ++k)
            cv[k] = Ops::LOADVPD(child + (std::size_t)k * npatt + h);
        for (j = 0; j < NS; ++j) {
            for (k = 0; k < NS; ++k)
                pv[k] = Ops::SETVPD(P[j * NS + k]);
            V acc = Ops::MULTVPD(cv[0], pv[0]);
            for (k = 1; k < NS; ++k)
                acc = Ops::FMADDVPD(cv[k], pv[k], acc);
            double* out = parent + (std::size_t)j * npatt + h;
            if (ACCUMULATE) acc = Ops::MULTVPD(Ops::LOADVPD(out), acc);
            Ops::STOREVPD(out, acc);
        }
    }

    if (h < npatt) {
        alignas(64) double cin[NS][W], cout[NS][W];
        const int rem = npatt - h;
        for (k = 0; k < NS; ++k) {
            for (j = 0; j < W; ++j) cin[k][j] = 0.0;
            std::memcpy(cin[k], child + (std::size_t)k * npatt + h,
                        (std::size_t)rem * sizeof(double));
        }
        for (j = 0; j < NS; ++j) {
            for (k = 0; k < W; ++k) cout[j][k] = 0.0;
            if (ACCUMULATE)
                std::memcpy(cout[j], parent + (std::size_t)j * npatt + h,
                            (std::size_t)rem * sizeof(double));
        }

        for (k = 0; k < NS; ++k) cv[k] = Ops::LOADVPD(cin[k]);
        for (j = 0; j < NS; ++j) {
            for (k = 0; k < NS; ++k)
                pv[k] = Ops::SETVPD(P[j * NS + k]);
            V acc = Ops::MULTVPD(cv[0], pv[0]);
            for (k = 1; k < NS; ++k)
                acc = Ops::FMADDVPD(cv[k], pv[k], acc);
            if (ACCUMULATE) acc = Ops::MULTVPD(Ops::LOADVPD(cout[j]), acc);
            Ops::STOREVPD(cout[j], acc);
        }
        for (j = 0; j < NS; ++j)
            std::memcpy(parent + (std::size_t)j * npatt + h, cout[j],
                        (std::size_t)rem * sizeof(double));
    }
}

/* Generic fallback for state counts without a specialisation.  Correct, but
   it pays the stack-tile penalty described above -- do not rely on it in a
   hot loop. */
template <bool ACCUMULATE, typename Ops>
void PmatMulGenericDouble(double* parent, const double* child, const double* P,
                          int nstates, int npatt) {
    using V = typename Ops::VECOBJ_PD;
    V cv[PYML_VEC_MAX_STATES], pv[PYML_VEC_MAX_STATES];
    int h = 0, j, k;

    for (; h + Ops::WIDTH_PD <= npatt; h += Ops::WIDTH_PD) {
        for (k = 0; k < nstates; ++k)
            cv[k] = Ops::LOADVPD(child + (std::size_t)k * npatt + h);
        for (j = 0; j < nstates; ++j) {
            for (k = 0; k < nstates; ++k)
                pv[k] = Ops::SETVPD(P[j * nstates + k]);
            V acc = Ops::MULTVPD(cv[0], pv[0]);
            for (k = 1; k < nstates; ++k)
                acc = Ops::FMADDVPD(cv[k], pv[k], acc);
            double* out = parent + (std::size_t)j * npatt + h;
            if (ACCUMULATE) acc = Ops::MULTVPD(Ops::LOADVPD(out), acc);
            Ops::STOREVPD(out, acc);
        }
    }
    for (; h < npatt; ++h)
        for (j = 0; j < nstates; ++j) {
            double s = P[j * nstates] * child[h];
            for (k = 1; k < nstates; ++k)
                s += P[j * nstates + k] * child[(std::size_t)k * npatt + h];
            double* out = parent + (std::size_t)j * npatt + h;
            *out = ACCUMULATE ? *out * s : s;
        }
}

template <bool ACCUMULATE, typename Ops>
void PmatMulGenericFloat(float* parent, const float* child, const float* P,
                         int nstates, int npatt) {
    using V = typename Ops::VECOBJ;
    V cv[PYML_VEC_MAX_STATES], pv[PYML_VEC_MAX_STATES];
    int h = 0, j, k;

    for (; h + Ops::WIDTH <= npatt; h += Ops::WIDTH) {
        for (k = 0; k < nstates; ++k)
            cv[k] = Ops::LOADVPS(child + (std::size_t)k * npatt + h);
        for (j = 0; j < nstates; ++j) {
            for (k = 0; k < nstates; ++k)
                pv[k] = Ops::SETVPS(P[j * nstates + k]);
            V acc = Ops::MULTVPS(cv[0], pv[0]);
            for (k = 1; k < nstates; ++k)
                acc = Ops::FMADDVPS(cv[k], pv[k], acc);
            float* out = parent + (std::size_t)j * npatt + h;
            if (ACCUMULATE) acc = Ops::MULTVPS(Ops::LOADVPS(out), acc);
            Ops::STOREVPS(out, acc);
        }
    }
    for (; h < npatt; ++h)
        for (j = 0; j < nstates; ++j) {
            float s = P[j * nstates] * child[h];
            for (k = 1; k < nstates; ++k)
                s += P[j * nstates + k] * child[(std::size_t)k * npatt + h];
            float* out = parent + (std::size_t)j * npatt + h;
            *out = ACCUMULATE ? *out * s : s;
        }
}

template <bool ACCUMULATE>
void PmatMulDispatchF64(double* parent, const double* child, const double* P,
                        int nstates, int npatt) {
    switch (nstates) {
        case 2:  PmatMulFixedDouble<2,  ACCUMULATE, DefaultOps>(parent, child, P, npatt); break;
        case 4:  PmatMulFixedDouble<4,  ACCUMULATE, DefaultOps>(parent, child, P, npatt); break;
        case 20: PmatMulFixedDouble<20, ACCUMULATE, DefaultOps>(parent, child, P, npatt); break;
        default:
            if (nstates > PYML_VEC_MAX_STATES) nstates = PYML_VEC_MAX_STATES;
            PmatMulGenericDouble<ACCUMULATE, DefaultOps>(parent, child, P, nstates, npatt);
            break;
    }
}

template <bool ACCUMULATE>
void PmatMulDispatchF32(float* parent, const float* child, const float* P,
                        int nstates, int npatt) {
    switch (nstates) {
        case 2:  PmatMulFixedFloat<2,  ACCUMULATE, DefaultOps>(parent, child, P, npatt); break;
        case 4:  PmatMulFixedFloat<4,  ACCUMULATE, DefaultOps>(parent, child, P, npatt); break;
        case 20: PmatMulFixedFloat<20, ACCUMULATE, DefaultOps>(parent, child, P, npatt); break;
        default:
            if (nstates > PYML_VEC_MAX_STATES) nstates = PYML_VEC_MAX_STATES;
            PmatMulGenericFloat<ACCUMULATE, DefaultOps>(parent, child, P, nstates, npatt);
            break;
    }
}

void PmatMulF64(double* parent, const double* child, const double* P,
                int nstates, long long npatt) {
    if (nstates <= 0 || npatt <= 0) return;
    PmatMulDispatchF64<false>(parent, child, P, nstates, (int)npatt);
}

void PmatMulAccF64(double* parent, const double* child, const double* P,
                   int nstates, long long npatt) {
    if (nstates <= 0 || npatt <= 0) return;
    PmatMulDispatchF64<true>(parent, child, P, nstates, (int)npatt);
}

void PmatMulF32(float* parent, const float* child, const float* P,
                int nstates, long long npatt) {
    if (nstates <= 0 || npatt <= 0) return;
    PmatMulDispatchF32<false>(parent, child, P, nstates, (int)npatt);
}

void PmatMulAccF32(float* parent, const float* child, const float* P,
                   int nstates, long long npatt) {
    if (nstates <= 0 || npatt <= 0) return;
    PmatMulDispatchF32<true>(parent, child, P, nstates, (int)npatt);
}

int VectorWidthFloat() noexcept { return DefaultOps::WIDTH; }
int VectorWidthDouble() noexcept { return DefaultOps::WIDTH_PD; }

const char* IsaName() noexcept {
#if defined(__AVX512F__)
    return "AVX-512F";
#elif defined(__AVX__)
    return "AVX";
#elif defined(__SSE2__)
    return "SSE2";
#else
    return "scalar";
#endif
}

bool HasFma() noexcept {
#if defined(__FMA__)
    return true;
#else
    return false;
#endif
}

template std::vector<float> Multiplier<float>(const float*, const float*, int);
template std::vector<double> Multiplier<double>(const double*, const double*, int);
template void MultiplyAccumulate<float>(float*, const float*, float, int);
template void MultiplyAccumulate<double>(double*, const double*, double, int);
template void MultiplyInPlace<float>(float*, const float*, int);
template void MultiplyInPlace<double>(double*, const double*, int);
template void ScaleInPlace<float>(float*, float, int);
template void ScaleInPlace<double>(double*, double, int);
}
