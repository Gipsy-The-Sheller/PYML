#include "vectorize_c.h"
#include "vectorize.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {

/// The C++ kernels take `int` lengths.  PAML's pattern counts are ints too,
/// so anything that does not fit is a caller bug rather than a real size.
int NarrowLength(long long n) noexcept {
    return static_cast<int>(
        std::min<long long>(n, std::numeric_limits<int>::max()));
}

}  // namespace

extern "C" {

int pyml_vec_width_f64(void) { return VecMnpUtils::VectorWidthDouble(); }
int pyml_vec_width_f32(void) { return VecMnpUtils::VectorWidthFloat(); }
const char* pyml_vec_isa_name(void) { return VecMnpUtils::IsaName(); }
int pyml_vec_has_fma(void) { return VecMnpUtils::HasFma() ? 1 : 0; }

/* dst = a * b, regardless of what dst held.  Aliasing dst == a or dst == b
 * is fine: the memcpy is then a no-op and the multiply is applied in place. */
void pyml_vec_mul_f64(double* dst, const double* a, const double* b, long long n) {
    if (n <= 0) return;
    const int len = NarrowLength(n);
    if (dst != a) std::memcpy(dst, a, static_cast<size_t>(len) * sizeof(double));
    VecMnpUtils::MultiplyInPlace(dst, b, len);
}

void pyml_vec_mul_f32(float* dst, const float* a, const float* b, long long n) {
    if (n <= 0) return;
    const int len = NarrowLength(n);
    if (dst != a) std::memcpy(dst, a, static_cast<size_t>(len) * sizeof(float));
    VecMnpUtils::MultiplyInPlace(dst, b, len);
}

void pyml_vec_muladd_f64(double* dst, const double* src, double coef, long long n) {
    if (n <= 0) return;
    VecMnpUtils::MultiplyAccumulate(dst, src, coef, NarrowLength(n));
}

void pyml_vec_muladd_f32(float* dst, const float* src, float coef, long long n) {
    if (n <= 0) return;
    VecMnpUtils::MultiplyAccumulate(dst, src, coef, NarrowLength(n));
}

void pyml_vec_scale_f64(double* dst, double coef, long long n) {
    if (n <= 0) return;
    VecMnpUtils::ScaleInPlace(dst, coef, NarrowLength(n));
}

void pyml_vec_scale_f32(float* dst, float coef, long long n) {
    if (n <= 0) return;
    VecMnpUtils::ScaleInPlace(dst, coef, NarrowLength(n));
}

void pyml_vec_fill_f64(double* dst, double value, long long n) {
    if (n <= 0) return;
    std::fill(dst, dst + NarrowLength(n), value);
}

void pyml_vec_fill_f32(float* dst, float value, long long n) {
    if (n <= 0) return;
    std::fill(dst, dst + NarrowLength(n), value);
}

void pyml_vec_pmat_mul_f64(double* parent, const double* child,
                           const double* P, int nstates, long long npatt) {
    VecMnpUtils::PmatMulF64(parent, child, P, nstates, npatt);
}

void pyml_vec_pmat_mul_f32(float* parent, const float* child,
                           const float* P, int nstates, long long npatt) {
    VecMnpUtils::PmatMulF32(parent, child, P, nstates, npatt);
}

void pyml_vec_pmat_mul_acc_f64(double* parent, const double* child,
                               const double* P, int nstates, long long npatt) {
    VecMnpUtils::PmatMulAccF64(parent, child, P, nstates, npatt);
}

void pyml_vec_pmat_mul_acc_f32(float* parent, const float* child,
                               const float* P, int nstates, long long npatt) {
    VecMnpUtils::PmatMulAccF32(parent, child, P, nstates, npatt);
}

double* pyml_vec_mul_new_f64(const double* a, const double* b, long long n) {
    if (n <= 0) return nullptr;
    const auto result = VecMnpUtils::Multiplier(a, b, NarrowLength(n));
    auto* out = static_cast<double*>(std::malloc(result.size() * sizeof(double)));
    if (out == nullptr) return nullptr;
    std::copy(result.begin(), result.end(), out);
    return out;
}

float* pyml_vec_mul_new_f32(const float* a, const float* b, long long n) {
    if (n <= 0) return nullptr;
    const auto result = VecMnpUtils::Multiplier(a, b, NarrowLength(n));
    auto* out = static_cast<float*>(std::malloc(result.size() * sizeof(float)));
    if (out == nullptr) return nullptr;
    std::copy(result.begin(), result.end(), out);
    return out;
}

void pyml_vec_free(void* p) { std::free(p); }

}  // extern "C"
