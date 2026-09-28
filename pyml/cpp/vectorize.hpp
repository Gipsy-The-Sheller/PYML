#ifndef PYML_CPP_VECTORIZE_HPP
#define PYML_CPP_VECTORIZE_HPP

#include <concepts>
#include <vector>

namespace VecMnpUtils {

template <typename T>
concept Vectorable = std::same_as<T, float> || std::same_as<T, double>;

template <Vectorable Precision>
std::vector<Precision> Multiplier(const Precision* queue1,
                                  const Precision* queue2,
                                  int qlength);

template <Vectorable Precision>
void MultiplyAccumulate(Precision* destination,
                        const Precision* values,
                        Precision coefficient,
                        int length);

/// In-place elementwise multiply,  destination[i] *= values[i].
/// Unlike Multiplier() this allocates nothing, so it is safe to call inside
/// a pruning hot loop.
template <Vectorable Precision>
void MultiplyInPlace(Precision* destination,
                     const Precision* values,
                     int length);

/// In-place shrink,  destination[i] *= coefficient.
template <Vectorable Precision>
void ScaleInPlace(Precision* destination,
                  Precision coefficient,
                  int length);

/// Maximum number of states pyml_vec_pmat_mul_*() can handle.
#define PYML_VEC_MAX_STATES 64

/// One Felsenstein partial:  parent[j][h] = sum_k P[j][k] * child[k][h].
///
/// Both partials are state-major with patterns contiguous -- parent[j*npatt+h]
/// and child[k*npatt+h] -- and P is row-major, P[j*nstates + k].
///
/// This is the kernel the pruning loop actually wants.  Expressing the same
/// update as nstates*nstates separate AXPYs (each streaming the whole child
/// block and rewriting the whole parent block) costs a factor of nstates more
/// memory traffic and is *slower* than the scalar code it replaces for the
/// small state counts phylogenetics uses.  Here each pattern tile is loaded
/// once, the reduction over k happens in registers, and the parent tile is
/// stored once -- the same traffic as the scalar loop, with nstates-wide SIMD
/// on the pattern axis.
void PmatMulF64(double* parent, const double* child,
                const double* P, int nstates, long long npatt);
void PmatMulF32(float* parent, const float* child,
                const float* P, int nstates, long long npatt);

/// Same, but multiplying into the existing parent:  parent *= sum_k P[j][k] *
/// child[k][h].  This is what a node's second and later children need, and
/// fusing the multiply into the same pass avoids an extra sweep over the
/// parent block.
void PmatMulAccF64(double* parent, const double* child,
                   const double* P, int nstates, long long npatt);
void PmatMulAccF32(float* parent, const float* child,
                   const float* P, int nstates, long long npatt);

/// Widest SIMD register this translation unit was compiled for.
/// 512 = AVX-512F, 256 = AVX, 128 = SSE2, 32 = scalar fallback.
int VectorWidthFloat() noexcept;
int VectorWidthDouble() noexcept;

/// "AVX-512F" / "AVX" / "SSE2" / "scalar"
const char* IsaName() noexcept;

/// Whether the build targeted FMA, i.e. whether MultiplyAccumulate() fuses.
bool HasFma() noexcept;

extern template std::vector<float> Multiplier<float>(
    const float*, const float*, int);
extern template std::vector<double> Multiplier<double>(
    const double*, const double*, int);
extern template void MultiplyInPlace<float>(float*, const float*, int);
extern template void MultiplyInPlace<double>(double*, const double*, int);
extern template void ScaleInPlace<float>(float*, float, int);
extern template void ScaleInPlace<double>(double*, double, int);

}  // namespace VecMnpUtils

#endif  // PYML_CPP_VECTORIZE_HPP
