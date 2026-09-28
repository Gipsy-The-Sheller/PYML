/* ===========================================================================
 *  vectorize_c.h — C ABI for pyml's SIMD vector kernels
 *
 *  vectorize.hpp is C++-only: it is template-based, and its Multiplier()
 *  returns a std::vector by value, so every call allocates.  Neither is
 *  usable from C code or from inside a pruning hot loop.  This header
 *  exposes the same kernels through a flat extern "C" surface with
 *  caller-owned buffers and no allocation at all.
 *
 *  All operations are elementwise over flat contiguous arrays of length n.
 *  f32 and f64 variants are provided; the f64 ones are what a likelihood
 *  pruner wants.
 *
 *  Build note: compile vectorize.cpp and vectorize_c.cpp with the same
 *  -march / -mavx2 / -mfma flags.  The instruction set actually used is
 *  fixed at compile time and can be queried with pyml_vec_isa_name().
 * =========================================================================== */

#ifndef PYML_VECTORIZE_C_H
#define PYML_VECTORIZE_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------- info --- */

/* Widest SIMD register compiled in: 512 (AVX-512F), 256 (AVX),
 * 128 (SSE2), or 32 (scalar fallback). */
int pyml_vec_width_f64(void);
int pyml_vec_width_f32(void);

/* "AVX-512F" / "AVX" / "SSE2" / "scalar" */
const char* pyml_vec_isa_name(void);

/* Non-zero if the translation unit was compiled with FMA, i.e. whether
 * pyml_vec_muladd_*() uses a single fused rounding step. */
int pyml_vec_has_fma(void);

/* ------------------------------------------------------------ arithmetic --- */

/* dst[i] = a[i] * b[i].  dst may alias a or b. */
void pyml_vec_mul_f64(double* dst, const double* a, const double* b, long long n);
void pyml_vec_mul_f32(float* dst, const float* a, const float* b, long long n);

/* dst[i] += coef * src[i]   -- the AXPY that drives the pruner.
 * This is the hot kernel: one broadcast coefficient, unit-stride src/dst. */
void pyml_vec_muladd_f64(double* dst, const double* src, double coef, long long n);
void pyml_vec_muladd_f32(float* dst, const float* src, float coef, long long n);

/* dst[i] *= coef */
void pyml_vec_scale_f64(double* dst, double coef, long long n);
void pyml_vec_scale_f32(float* dst, float coef, long long n);

/* dst[i] = value */
void pyml_vec_fill_f64(double* dst, double value, long long n);
void pyml_vec_fill_f32(float* dst, float value, long long n);

/* ---------------------------------------------------- partial likelihood --- */

/* One Felsenstein partial:
 *     parent[j*npatt + h] = sum_k P[j*nstates + k] * child[k*npatt + h]
 * for j,k in [0,nstates) and h in [0,npatt).  Both partials are state-major
 * with patterns contiguous; P is row-major.  nstates must be <= 64.
 *
 * This is the kernel a pruning loop wants.  Writing the same update as
 * nstates*nstates separate muladd calls re-streams both partials once per
 * (j,k) pair -- nstates times the traffic for the same arithmetic. */
void pyml_vec_pmat_mul_f64(double* parent, const double* child,
                           const double* P, int nstates, long long npatt);
void pyml_vec_pmat_mul_f32(float* parent, const float* child,
                           const float* P, int nstates, long long npatt);

/* Same, multiplied into the existing parent:  parent *= sum_k P[j][k]*child.
 * Use this for a node's second and later children; it saves a whole extra
 * sweep over the parent block. */
void pyml_vec_pmat_mul_acc_f64(double* parent, const double* child,
                               const double* P, int nstates, long long npatt);
void pyml_vec_pmat_mul_acc_f32(float* parent, const float* child,
                               const float* P, int nstates, long long npatt);

/* ------------------------------------------------------------ allocating --- */
/* Parity with the C++ Multiplier().  Not for hot loops -- these allocate.
 * Free the result with pyml_vec_free(). */

double* pyml_vec_mul_new_f64(const double* a, const double* b, long long n);
float* pyml_vec_mul_new_f32(const float* a, const float* b, long long n);
void pyml_vec_free(void* p);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PYML_VECTORIZE_C_H */
