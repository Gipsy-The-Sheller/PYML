/* Rig for pyml/cpp/vectorize_c.h — exercises the C ABI the way a C caller
 * (e.g. the PAML patch) uses it, and checks every kernel against a naive
 * scalar loop.  Lengths deliberately include values that are not multiples
 * of any SIMD width, so the tail handling is covered too.
 *
 *   gcc  -O2 -c test_vectorize_c.c
 *   g++  -O2 -mavx2 -mfma -o t test_vectorize_c.o vectorize.o vectorize_c.o -lm
 */

#include "vectorize_c.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

static void check(int cond, const char* what) {
    if (!cond) {
        printf("  FAIL  %s\n", what);
        ++failures;
    }
}

/* Deterministic pseudo-random doubles in [0.1, 1.1). */
static double sample(int i) {
    unsigned x = (unsigned)i * 2654435761u;
    x ^= x >> 13;
    x *= 2246822519u;
    x ^= x >> 16;
    return 0.1 + 1.0 * ((double)(x % 10000) / 10000.0);
}

static void test_mul(long long n) {
    double* a = malloc(n * sizeof(double));
    double* b = malloc(n * sizeof(double));
    double* got = malloc(n * sizeof(double));
    for (long long i = 0; i < n; ++i) { a[i] = sample((int)i); b[i] = sample((int)(i + n)); }

    pyml_vec_mul_f64(got, a, b, n);
    for (long long i = 0; i < n; ++i)
        if (fabs(got[i] - a[i] * b[i]) > 1e-15 * fabs(a[i] * b[i])) {
            printf("  FAIL  mul_f64 n=%lld at %lld\n", n, i); ++failures; break;
        }

    /* aliasing: dst == a must behave identically */
    for (long long i = 0; i < n; ++i) got[i] = a[i];
    pyml_vec_mul_f64(got, got, b, n);
    for (long long i = 0; i < n; ++i)
        if (fabs(got[i] - a[i] * b[i]) > 1e-15 * fabs(a[i] * b[i])) {
            printf("  FAIL  mul_f64 aliased n=%lld at %lld\n", n, i); ++failures; break;
        }

    free(a); free(b); free(got);
}

static void test_muladd(long long n) {
    double* dst = malloc(n * sizeof(double));
    double* src = malloc(n * sizeof(double));
    double* want = malloc(n * sizeof(double));
    const double coef = -1.3;
    for (long long i = 0; i < n; ++i) {
        dst[i] = want[i] = sample((int)i) * 0.5;
        src[i] = sample((int)(i + n));
    }

    pyml_vec_muladd_f64(dst, src, coef, n);
    for (long long i = 0; i < n; ++i) {
        const double orig = want[i];
        want[i] += coef * src[i];
        /* FMA rounds once where the scalar loop rounds twice.  The two answers
         * are both correct; they may differ by one ulp of the OPERANDS, which
         * is far larger than an ulp of the result when the sum cancels.
         * Bound against the operand magnitudes, not the result. */
        const double bound = 4.0 * 2.220446049250313e-16 * (fabs(orig) + fabs(coef * src[i]));
        if (fabs(dst[i] - want[i]) > bound) {
            printf("  FAIL  muladd_f64 n=%lld at %lld: %.17g vs %.17g\n",
                   n, i, dst[i], want[i]);
            ++failures; break;
        }
    }
    free(dst); free(src); free(want);
}

static void test_scale_and_fill(long long n) {
    double* v = malloc(n * sizeof(double));
    pyml_vec_fill_f64(v, 2.5, n);
    for (long long i = 0; i < n; ++i) check(v[i] == 2.5, "fill");
    pyml_vec_scale_f64(v, 2.0, n);
    for (long long i = 0; i < n; ++i) check(v[i] == 5.0, "scale");
    free(v);
}

static void test_mul_new(long long n) {
    double* a = malloc(n * sizeof(double));
    double* b = malloc(n * sizeof(double));
    for (long long i = 0; i < n; ++i) { a[i] = sample((int)i); b[i] = sample((int)(i + n)); }
    double* got = pyml_vec_mul_new_f64(a, b, n);
    check(got != NULL, "mul_new returns a buffer");
    if (got) {
        for (long long i = 0; i < n; ++i)
            if (fabs(got[i] - a[i] * b[i]) > 1e-15 * fabs(a[i] * b[i])) {
                printf("  FAIL  mul_new n=%lld at %lld\n", n, i); ++failures; break;
            }
        pyml_vec_free(got);
    }
    free(a); free(b);
}

/* parent[j][h] = sum_k P[j][k] * child[k][h], naive.  Both partials are
   state-major with npatt patterns contiguous. */
static void naive_pmat_mul(double* parent, const double* child, const double* P,
                           int n, long long npatt, int accumulate)
{
    for (long long h = 0; h < npatt; ++h)
        for (int j = 0; j < n; ++j) {
            double s = 0;
            for (int k = 0; k < n; ++k)
                s += P[j * n + k] * child[(size_t)k * npatt + h];
            double* out = parent + (size_t)j * npatt + h;
            *out = accumulate ? *out * s : s;
        }
}

static void test_pmat_mul(int n, long long npatt) {
    const size_t blk = (size_t)n * npatt;
    double* child = malloc(blk * sizeof(double));
    double* P = malloc((size_t)n * n * sizeof(double));
    double* got = malloc(blk * sizeof(double));
    double* want = malloc(blk * sizeof(double));

    for (size_t i = 0; i < blk; ++i) child[i] = sample((int)i) * 0.7;
    for (int i = 0; i < n * n; ++i) P[i] = sample(1000 + i) * 0.3;

    for (int mode = 0; mode < 2; ++mode) {
        for (size_t i = 0; i < blk; ++i) got[i] = want[i] = sample((int)i) * 0.5;
        if (mode) pyml_vec_pmat_mul_acc_f64(got, child, P, n, npatt);
        else      pyml_vec_pmat_mul_f64(got, child, P, n, npatt);
        naive_pmat_mul(want, child, P, n, npatt, mode);

        for (size_t i = 0; i < blk; ++i) {
            const double bound = 4.0 * 2.220446049250313e-16 *
                                 (fabs(want[i]) + fabs(got[i]));
            if (fabs(got[i] - want[i]) > bound) {
                printf("  FAIL  pmat_mul%s n=%d npatt=%lld at %lld: "
                       "%.17g vs %.17g\n", mode ? "_acc" : "", n, npatt,
                       (long long)i, got[i], want[i]);
                ++failures;
                goto done;
            }
        }
    }
done:
    free(child); free(P); free(got); free(want);
}

int main(void) {
    printf("ISA          : %s\n", pyml_vec_isa_name());
    printf("f64 width    : %d doubles per register\n", pyml_vec_width_f64());
    printf("f32 width    : %d floats per register\n", pyml_vec_width_f32());
    printf("FMA          : %s\n", pyml_vec_has_fma() ? "yes (fused)" : "no (mul+add)");

    /* 1,2,3,4 cover every SIMD width and the sub-width scalar tail */
    static const long long lengths[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 1000};
    for (size_t k = 0; k < sizeof(lengths) / sizeof(lengths[0]); ++k) {
        const long long n = lengths[k];
        test_mul(n);
        test_muladd(n);
        if (n > 0) { test_scale_and_fill(n); test_mul_new(n); }
    }

    /* pmat_mul: every tail remainder 0..W-1, at each specialised state
       count, in both the overwrite and accumulate modes */
    static const long long patts[] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 15, 16, 17, 100, 1001};
    for (size_t k = 0; k < sizeof(patts) / sizeof(patts[0]); ++k)
        for (int n = 2; n <= 4; n += 2)
            test_pmat_mul(n, patts[k]);

    printf(failures ? "\n%d FAILURES\n" : "\nall kernels OK\n", failures);
    return failures != 0;
}
