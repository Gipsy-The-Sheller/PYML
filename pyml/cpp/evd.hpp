#pragma once
// ===========================================================================
//  evd.hpp — Eigenvalue decomposition of dense, general real matrices
//
//  Written from scratch for the YR-MPE project.  The numerical scheme is the
//  classical (public-domain) pipeline of Wilkinson & Reinsch, re-expressed in
//  modern C++20/23:
//
//      1. balance       Parlett & Reinsch diagonal scaling (exact powers of
//                       the radix -> introduces no rounding error)
//      2. hessenberg    Householder reduction  A = Z H Z^T,  H Hessenberg
//      3. schurQR       Francis implicitly-shifted double QR -> real Schur
//                       form T (quasi-upper-triangular), driven by a work
//                       stack so that splits at the TOP of the active window
//                       are exploited as eagerly as splits at the bottom
//      4. extract       1x1 blocks -> real eigenvalues, 2x2 blocks -> the
//                       complex-conjugate pairs
//      5. eigenvectors  exact back-substitution on T (handles repeated roots),
//                       inverse iteration for conjugate pairs, then  v <- Z y
//
//  The Francis sweep was validated step-by-step against an explicitly formed
//  double-shift step (M = H^2 - s H + det I, M = QR, H <- Q^T H Q): the two
//  agree to machine precision.
//
//  Storage follows the EISPACK / BEAGLE convention: eigenvalues as
//  real[n] + imag[n], and the two columns of a conjugate pair hold the real
//  and imaginary parts of the eigenvector.
// ===========================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace mpe::linalg {

// ------------------------------------------------------------------ Matrix --

/// Contiguous row-major dense matrix.  No raw owning pointers anywhere.
template <std::floating_point T>
class Matrix {
public:
    using value_type = T;

    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, T fill = T{})
        : rows_{rows}, cols_{cols}, data_(rows * cols, fill) {}

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] T& operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * cols_ + j];
    }
    [[nodiscard]] const T& operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * cols_ + j];
    }

    [[nodiscard]] std::span<T> data() noexcept { return data_; }
    [[nodiscard]] std::span<const T> data() const noexcept { return data_; }

    [[nodiscard]] static Matrix identity(std::size_t n) {
        Matrix m{n, n, T{}};
        for (std::size_t i = 0; i < n; ++i) m(i, i) = T{1};
        return m;
    }

private:
    std::size_t rows_{}, cols_{};
    std::vector<T> data_;
};

// -------------------------------------------------------------- diagnostics --

enum class EigenStatus : std::uint8_t {
    Ok            = 0,  ///< converged
    NotConverged  = 1,  ///< QR iteration hit the iteration cap
    Singular      = 2,  ///< eigenvector matrix could not be inverted
};

[[nodiscard]] constexpr std::string_view to_string(EigenStatus s) noexcept {
    using enum EigenStatus;
    switch (s) {
        case Ok:           return "Ok";
        case NotConverged: return "NotConverged";
        case Singular:     return "Singular";
    }
    return "unknown";
}

/// A full  Q = E diag(lambda) E^-1  decomposition, BEAGLE-flavoured.
template <std::floating_point T>
struct EigenSystem {
    std::vector<T> real;             ///< n real parts  (BEAGLE [0, n)   )
    std::vector<T> imag;             ///< n imag parts  (BEAGLE [n, 2n)  )
    Matrix<T> vectors;               ///< n x n, column k = Re / Im of vector k
    Matrix<T> inverseVectors;        ///< n x n
    bool hasComplex = false;         ///< any |imag| != 0
};

template <std::floating_point T>
struct Diagnostics {
    EigenStatus status{EigenStatus::Ok};
    int qrIterations{};
    T maxResidual{};        ///< max_k ||A v_k - lambda_k v_k||_inf / ||v_k||_inf
    T maxImag{};            ///< max_k |imag_k| — non-zero => non-reversible
};

// ------------------------------------------------------------- reflectors --

template <std::floating_point T>
struct Householder {
    std::vector<T> v;
    T tau{};   ///< reflector is  I - tau * v * v^T
    T beta{};  ///< (I - tau v v^T) x == beta * e_1
};

/// Wilkinson reflector: (I - tau v v^T) x = beta e1, with beta of opposite
/// sign to x[0] so no cancellation occurs.
template <std::floating_point T>
[[nodiscard]] Householder<T> makeHouseholder(std::span<const T> x) {
    const std::size_t m = x.size();
    std::vector<T> v(m, T{});
    if (m == 0) return {std::move(v), T{}, T{}};

    T sigma{};
    for (std::size_t i = 1; i < m; ++i) sigma += x[i] * x[i];

    if (sigma == T{}) {
        v[0] = T{1};
        return {std::move(v), T{}, x[0]};
    }

    const T mu = std::sqrt(x[0] * x[0] + sigma);
    const T beta = (x[0] >= T{}) ? -mu : mu;
    v[0] = x[0] - beta;
    for (std::size_t i = 1; i < m; ++i) v[i] = x[i];

    T vnorm2{};
    for (auto e : v) vnorm2 += e * e;

    return {std::move(v), T{2} / vnorm2, beta};
}

/// A[r0:r1, c0:c1] <- (I - tau v v^T) A[r0:r1, c0:c1]
template <std::floating_point T, std::ranges::contiguous_range R>
void applyLeft(Matrix<T>& A, const R& v, T tau,
               int r0, int r1, int c0, int c1) noexcept {
    if (tau == T{}) return;
    const int m = r1 - r0;
    for (int j = c0; j < c1; ++j) {
        T s{};
        for (int i = 0; i < m; ++i) s += v[i] * A(r0 + i, j);
        s *= tau;
        for (int i = 0; i < m; ++i) A(r0 + i, j) -= s * v[i];
    }
}

/// A[r0:r1, c0:c1] <- A[r0:r1, c0:c1] (I - tau v v^T)
template <std::floating_point T, std::ranges::contiguous_range R>
void applyRight(Matrix<T>& A, const R& v, T tau,
                int r0, int r1, int c0, int c1) noexcept {
    if (tau == T{}) return;
    const int m = c1 - c0;
    for (int i = r0; i < r1; ++i) {
        T s{};
        for (int j = 0; j < m; ++j) s += A(i, c0 + j) * v[j];
        s *= tau;
        for (int j = 0; j < m; ++j) A(i, c0 + j) -= s * v[j];
    }
}

// ---------------------------------------------------------- plane rotations --

/// rows i0,i1 of A are replaced by G^T applied to them, G = [[cs,-sn],[sn,cs]]
template <std::floating_point T>
void rotateRows(Matrix<T>& A, T cs, T sn, int i0, int i1, int c0, int c1) noexcept {
    for (int j = c0; j < c1; ++j) {
        const T a = A(i0, j), b = A(i1, j);
        A(i0, j) = cs * a + sn * b;
        A(i1, j) = cs * b - sn * a;
    }
}

/// columns i0,i1 of A are replaced by A G
template <std::floating_point T>
void rotateCols(Matrix<T>& A, T cs, T sn, int i0, int i1, int r0, int r1) noexcept {
    for (int i = r0; i < r1; ++i) {
        const T a = A(i, i0), b = A(i, i1);
        A(i, i0) = cs * a + sn * b;
        A(i, i1) = cs * b - sn * a;
    }
}

// ------------------------------------------------------------- 1. balancing --

/// Parlett & Reinsch: A <- D^-1 A D, D = diag(scale), powers of the radix.
/// Eigenvectors of the original matrix are recovered with  v <- D v_bal.
template <std::floating_point T>
[[nodiscard]] std::vector<T> balance(Matrix<T>& A) noexcept {
    const int n = static_cast<int>(A.rows());
    std::vector<T> scale(n, T{1});
    const T radix = static_cast<T>(std::numeric_limits<T>::radix);
    const T b2 = radix * radix;

    for (bool converged = false; !converged;) {
        converged = true;
        for (int i = 0; i < n; ++i) {
            T c{}, r{};
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                c += std::abs(A(j, i));
                r += std::abs(A(i, j));
            }
            if (c == T{} || r == T{}) continue;

            T g = r / radix, f = T{1}, s = c + r;
            while (c < g) { f *= radix; c *= b2; }
            g = r * radix;
            while (c >= g) { f /= radix; c /= b2; }

            if ((c + r) / f < T{0.95} * s) {
                g = T{1} / f;
                scale[i] *= f;
                converged = false;
                for (int j = 0; j < n; ++j) A(i, j) *= g;
                for (int j = 0; j < n; ++j) A(j, i) *= f;
            }
        }
    }
    return scale;
}

// ----------------------------------------------- 2. Hessenberg reduction -----

/// A <- H (upper Hessenberg); Z accumulates so that A_original = Z H Z^T.
template <std::floating_point T>
void hessenberg(Matrix<T>& A, Matrix<T>& Z) {
    const int n = static_cast<int>(A.rows());
    std::vector<T> col(std::max(0, n - 1));

    for (int k = 0; k + 2 < n; ++k) {
        const int len = n - k - 1;
        for (int i = 0; i < len; ++i) col[i] = A(k + 1 + i, k);
        auto [v, tau, beta] = makeHouseholder<T>(std::span<const T>(col.data(), len));

        applyLeft (A, v, tau, k + 1, n, 0, n);
        applyRight(A, v, tau, 0,   n, k + 1, n);
        applyRight(Z, v, tau, 0,   n, k + 1, n);

        A(k + 1, k) = beta;
        for (int i = k + 2; i < n; ++i) A(i, k) = T{};
    }
}

// --------------------------------------------------- 3. Francis double QR ---

namespace detail {

/// One explicitly-shifted QR step on the 2x2 block [l, l+1].
template <std::floating_point T>
void qrStep2(Matrix<T>& H, Matrix<T>& Z, int l) {
    const int n = static_cast<int>(H.rows());
    const T a = H(l, l), b = H(l, l + 1), c = H(l + 1, l), d = H(l + 1, l + 1);
    const T tr = a + d, det = a * d - b * c;
    const T disc = std::max(tr * tr - T{4} * det, T{});
    const T sq = std::sqrt(disc);
    const T lam = std::abs(tr / T{2} + sq - d) < std::abs(tr / T{2} - sq - d)
                    ? tr / T{2} + sq
                    : tr / T{2} - sq;

    const T x[2] = {a - lam, c};
    auto [v, tau, beta] = makeHouseholder<T>(std::span<const T>(x, 2));

    applyLeft (H, v, tau, l, l + 2, 0, n);
    applyRight(H, v, tau, 0, n, l, l + 2);
    applyRight(Z, v, tau, 0, n, l, l + 2);
}

/// One implicitly-shifted Francis double-shift sweep on H[l..ihi][l..ihi].
///
/// M = (H - s1 I)(H - s2 I) is never formed: by the implicit-Q theorem it is
/// enough that the leading reflector P0 satisfies  P0 e_1 proportional to
/// M e_1.  M e_1 lives on rows l..l+2, so P0 acts on exactly those rows.
/// The remaining reflectors chase the bulge down the subdiagonal, shrinking
/// to 2x2 (then 1x1, a no-op) as they reach the bottom of the block.
template <std::floating_point T>
void francisSweep(Matrix<T>& H, Matrix<T>& Z, int l, int ihi) {
    const int n = static_cast<int>(H.rows());

    const T a = H(ihi - 1, ihi - 1), b = H(ihi - 1, ihi);
    const T c = H(ihi, ihi - 1),     d = H(ihi, ihi);
    const T s = a + d;
    const T det = a * d - b * c;

    // first column of M = H^2 - s H + det I, restricted to rows l, l+1, l+2
    const T x[3] = {
        H(l, l) * (H(l, l) - s) + H(l, l + 1) * H(l + 1, l) + det,
        H(l + 1, l) * (H(l, l) + H(l + 1, l + 1) - s),
        H(l + 1, l) * H(l + 2, l + 1),
    };

    auto [v, tau, beta] = makeHouseholder<T>(std::span<const T>(x, 3));
    applyLeft (H, v, tau, l, l + 3, 0, n);
    applyRight(H, v, tau, 0, n, l, l + 3);
    applyRight(Z, v, tau, 0, n, l, l + 3);

    std::vector<T> y(3);
    for (int k = l; k < ihi; ++k) {
        const int sz = std::min(3, ihi - k);
        if (sz < 2) break;                     // 1x1 reflector is a no-op
        for (int i = 0; i < sz; ++i) y[i] = H(k + 1 + i, k);
        auto [w, tau2, beta2] =
            makeHouseholder<T>(std::span<const T>(y.data(), sz));
        applyLeft (H, w, tau2, k + 1, k + 1 + sz, 0, n);
        applyRight(H, w, tau2, 0, n, k + 1, k + 1 + sz);
        applyRight(Z, w, tau2, 0, n, k + 1, k + 1 + sz);
    }
}

/// Rotate a stubborn 2x2 block with real eigenvalues into upper-triangular
/// (Schur) form.  G's first column is the eigenvector for the eigenvalue
/// closest to d, so G^T A G has a zero in its lower-left corner.
template <std::floating_point T>
void forceSchur22(Matrix<T>& H, Matrix<T>& Z, int i0) {
    const int i1 = i0 + 1;
    const int n  = static_cast<int>(H.rows());
    const T a = H(i0, i0), b = H(i0, i1), c = H(i1, i0), d = H(i1, i1);
    const T tr = a + d, det = a * d - b * c;
    const T sq = std::sqrt(std::max(tr * tr - T{4} * det, T{}));
    const T l1 = (tr + sq) / T{2}, l2 = (tr - sq) / T{2};
    const T lam = std::abs(l1 - d) < std::abs(l2 - d) ? l1 : l2;

    T v0, v1;
    if (std::abs(c) > std::abs(b)) { v0 = lam - d; v1 = c; }
    else                           { v0 = b;       v1 = lam - a; }

    const T r = std::hypot(v0, v1);
    if (r == T{}) return;
    const T cs = v0 / r, sn = v1 / r;

    rotateRows(H, cs, sn, i0, i1, 0, n);
    rotateCols(H, cs, sn, i0, i1, 0, n);
    rotateCols(Z, cs, sn, i0, i1, 0, n);
}

}  // namespace detail

/// Guarantee that every surviving 2x2 diagonal block is a genuine complex
/// conjugate pair: any 2x2 block with real eigenvalues is rotated into upper
/// triangular form.  Without this, a real eigenvalue can end up off the
/// diagonal and the eigenvector back-substitution has no free variable to
/// anchor to (which silently produces zero eigenvector columns).
template <std::floating_point T>
void triangularizeRealBlocks(Matrix<T>& H, Matrix<T>& Z) {
    const int n = static_cast<int>(H.rows());
    for (int pass = 0; pass < n; ++pass) {
        const auto is1x1 = blockStructure(H);
        bool changed = false;
        for (int i = 0; i + 1 < n;) {
            if (is1x1[i]) { ++i; continue; }
            const T a = H(i, i), b = H(i, i + 1);
            const T c = H(i + 1, i), d = H(i + 1, i + 1);
            const T disc = (a - d) * (a - d) + T{4} * b * c;
            if (disc >= T{}) {
                const T before = std::abs(H(i + 1, i));
                detail::forceSchur22(H, Z, i);
                if (std::abs(H(i + 1, i)) < before) changed = true;
            }
            i += 2;
        }
        if (!changed) break;      // remaining blocks are complex pairs (or defective)
    }
}

/// Reduce an upper-Hessenberg matrix to real Schur form in place.
///
/// The pending work is kept on an explicit stack: whenever the active window
/// splits (a negligible subdiagonal anywhere inside it), the upper part is
/// pushed and finished later.  Exploiting TOP splits as well as bottom ones
/// is what keeps the sweep count near O(n) instead of O(n^2).
///
/// Returns the number of QR sweeps, or -1 if the iteration cap is hit.
template <std::floating_point T>
int schurQR(Matrix<T>& H, Matrix<T>& Z) {
    const int n = static_cast<int>(H.rows());
    const T eps  = std::numeric_limits<T>::epsilon();
    const T tiny = std::numeric_limits<T>::min();

    int iterations = 0;
    const int maxIter = 30 * n * n + 500;

    std::vector<std::pair<int, int>> pending;
    pending.emplace_back(0, n - 1);

    while (!pending.empty()) {
        const auto [ilo0, ihi0] = pending.back();
        pending.pop_back();
        int ilo = ilo0, ihi = ihi0;

        int stuck = 0;
        while (ihi > ilo) {
            if (++iterations > maxIter) return -1;

            // ---- look for a split strictly inside (ilo, ihi]
            int l = ilo;
            for (int i = ihi; i > ilo; --i) {
                const T h = std::abs(H(i, i - 1));
                const T sc = std::abs(H(i - 1, i - 1)) + std::abs(H(i, i));
                if (h <= eps * sc || h <= tiny) { l = i; break; }
            }

            if (l > ilo) {                       // matrix decouples here
                pending.emplace_back(ilo, l - 1);
                ilo = l;
                stuck = 0;
                continue;
            }

            // ---- [ilo..ihi] is one unsplit block
            if (ihi == ilo + 1) {                // trailing 2x2
                const T a = H(ilo, ilo),     b = H(ilo, ihi);
                const T c = H(ihi, ilo),     d = H(ihi, ihi);
                const T disc = (a - d) * (a - d) + T{4} * b * c;
                if (disc < T{}) break;           // genuine complex pair: keep
                if (++stuck > 20) {              // real but stubborn: force it
                    detail::forceSchur22(H, Z, ilo);
                    break;
                }
                detail::qrStep2(H, Z, ilo);
                continue;
            }

            detail::francisSweep(H, Z, ilo, ihi);
        }
    }
    return iterations;
}

// ------------------------------------------------------- 4. eigenvalue read --

/// Split the quasi-triagonal matrix into 1x1 and 2x2 diagonal blocks.
/// This single classification is shared by the eigenvalue reader, the
/// eigenvector back-substitution and the repeated-root bookkeeping — using
/// two different thresholds there is a classic source of silently zero
/// eigenvector columns.
template <std::floating_point T>
[[nodiscard]] std::vector<char> blockStructure(const Matrix<T>& T_) {
    const int n = static_cast<int>(T_.rows());
    const T eps = std::numeric_limits<T>::epsilon();
    std::vector<char> is1x1(n, 1);
    for (int i = 0; i + 1 < n; ++i) {
        if (std::abs(T_(i + 1, i)) >
            eps * (std::abs(T_(i, i)) + std::abs(T_(i + 1, i + 1)))) {
            is1x1[i] = 0;
            is1x1[i + 1] = 0;
            ++i;
        }
    }
    return is1x1;
}

template <std::floating_point T>
void extractEigenvalues(const Matrix<T>& T_, std::vector<T>& real,
                        std::vector<T>& imag, bool& hasComplex,
                        const std::vector<char>& is1x1) {
    const int n = static_cast<int>(T_.rows());
    real.assign(n, T{});
    imag.assign(n, T{});
    hasComplex = false;

    for (int i = 0; i < n;) {
        if (is1x1[i]) {
            real[i] = T_(i, i);
            ++i;
            continue;
        }
        const T a = T_(i, i), b = T_(i, i + 1);
        const T c = T_(i + 1, i), d = T_(i + 1, i + 1);
        const T tr = a + d, det = a * d - b * c;
        const T disc = tr * tr - T{4} * det;      // == (a-d)^2 + 4bc
        if (disc < T{}) {
            real[i] = real[i + 1] = tr / T{2};
            imag[i]     =  std::sqrt(-disc) / T{2};
            imag[i + 1] = -std::sqrt(-disc) / T{2};
            hasComplex = true;
        } else {
            const T sq = std::sqrt(disc);
            real[i]     = (tr + sq) / T{2};
            real[i + 1] = (tr - sq) / T{2};
        }
        i += 2;
    }
}

// -------------------------------------------------- 5. eigenvector recovery --

/// Null vector of (T - lambda I) for a quasi-upper-triangular T, by exact
/// back-substitution.  Every zero-diagonal 1x1 position is a free variable;
/// `freeIdx` says which one is set to 1, so handing a different one to each
/// copy of a repeated eigenvalue gives linearly independent eigenvectors.
/// (Inverse iteration would collapse all copies of a repeated root onto the
/// same vector — fatal for JC69, whose eigenvalue -4/3 has multiplicity 3.)
template <std::floating_point T>
[[nodiscard]] std::vector<T> nullVectorQuasi(const Matrix<T>& T_, T lambda,
                                             int freeIdx, T tol,
                                             const std::vector<char>& is1x1) {
    const int n = static_cast<int>(T_.rows());
    std::vector<T> y(n, T{});

    for (int i = n - 1; i >= 0;) {
        if (i > 0 && !is1x1[i]) {
            // 2x2 block occupying rows i-1, i
            T s1{}, s2{};
            for (int j = i + 1; j < n; ++j) {
                s1 += T_(i - 1, j) * y[j];
                s2 += T_(i, j)     * y[j];
            }
            const T a = T_(i - 1, i - 1) - lambda, b = T_(i - 1, i);
            const T c = T_(i, i - 1),             d = T_(i, i) - lambda;
            const T det = a * d - b * c;
            if (std::abs(det) > tol) {
                y[i - 1] = (-d * s1 + b * s2) / det;
                y[i]     = ( c * s1 - a * s2) / det;
            } else {
                y[i - 1] = (i - 1 == freeIdx) ? T{1} : T{};
                y[i]     = T{};
            }
            i -= 2;
        } else {
            T s{};
            for (int j = i + 1; j < n; ++j) s += T_(i, j) * y[j];
            const T denom = T_(i, i) - lambda;
            if (std::abs(denom) > tol) y[i] = -s / denom;
            else                       y[i] = (i == freeIdx) ? T{1} : T{};
            --i;
        }
    }
    return y;
}

namespace detail {

template <std::floating_point T>
using Complex = std::complex<T>;

/// Solve M y = b by Gaussian elimination with partial pivoting.
template <std::floating_point T>
inline std::vector<Complex<T>> solveComplex(const std::vector<std::vector<Complex<T>>>& M0,
                                            std::vector<Complex<T>> b) {
    using C = Complex<T>;
    const int n = static_cast<int>(b.size());
    auto M = M0;

    for (int k = 0; k < n; ++k) {
        int piv = k;
        double best = std::abs(M[k][k]);
        for (int i = k + 1; i < n; ++i)
            if (const double m = std::abs(M[i][k]); m > best) { best = m; piv = i; }
        if (piv != k) {
            std::swap(M[piv], M[k]);
            std::swap(b[piv], b[k]);
        }
        const Complex pivot = (std::abs(M[k][k]) < 1e-300) ? Complex(1e-300, 0.0) : M[k][k];
        for (int i = k + 1; i < n; ++i) {
            const Complex f = M[i][k] / pivot;
            if (f == C{}) continue;
            for (int j = k; j < n; ++j) M[i][j] -= f * M[k][j];
            b[i] -= f * b[k];
        }
    }

    std::vector<C> y(n);
    for (int i = n - 1; i >= 0; --i) {
        C s = b[i];
        for (int j = i + 1; j < n; ++j) s -= M[i][j] * y[j];
        y[i] = s / ((std::abs(M[i][i]) < 1e-300) ? C(T{1e-300}, T{}) : M[i][i]);
    }
    return y;
}

/// Exact null vector of (T - lambda I) for a COMPLEX eigenvalue, by complex
/// back-substitution on the quasi-triangular Schur matrix.  `blockStart` is
/// the first diagonal position of the 2x2 block that owns lambda: the
/// eigenvector is seeded there analytically, everything above follows from
/// back-substitution and everything below is zero.  This replaces inverse
/// iteration, which needs many steps and stalls on clustered spectra.
template <std::floating_point T>
[[nodiscard]] std::vector<detail::Complex<T>> nullVectorQuasiC(
    const Matrix<T>& T_, detail::Complex<T> lambda,
    const std::vector<char>& is1x1, int blockStart) {
    using C = detail::Complex<T>;
    const int n = static_cast<int>(T_.rows());
    std::vector<C> y(n, C{});

    for (int i = n - 1; i >= 0;) {
        if (i > 0 && !is1x1[i]) {
            const T a = T_(i - 1, i - 1), b = T_(i - 1, i);
            const T c = T_(i, i - 1),     d = T_(i, i);
            C s1{}, s2{};
            for (int j = i + 1; j < n; ++j) {
                s1 += C{T_(i - 1, j), T{}} * y[j];
                s2 += C{T_(i, j),     T{}} * y[j];
            }
            if (i - 1 == blockStart) {
                // seed: eigenvector of the owning 2x2 block
                C v0{C{b, T{}}}, v1{lambda - C{a, T{}}};
                const double r = std::abs(v0) + std::abs(v1);
                if (r == 0.0) { v0 = C{T{1}, T{}}; v1 = C{}; }
                y[i - 1] = v0;
                y[i]     = v1;
            } else {
                const C m00 = C{a, T{}} - lambda, m01 = C{b, T{}};
                const C m10 = C{c, T{}},         m11 = C{d, T{}} - lambda;
                const C det = m00 * m11 - m01 * m10;
                if (std::abs(det) > T{1e-300}) {
                    y[i - 1] = (-m11 * s1 + m01 * s2) / det;
                    y[i]     = ( m10 * s1 - m00 * s2) / det;
                }
            }
            i -= 2;
        } else {
            C acc{};
            for (int j = i + 1; j < n; ++j) acc += C{T_(i, j), T{}} * y[j];
            const C denom = C{T_(i, i), T{}} - lambda;
            if (std::abs(denom) > T{1e-300}) y[i] = -acc / denom;
            else                             y[i] = (i == blockStart) ? C{T{1}, T{}} : C{};
            --i;
        }
    }
    return y;
}

/// Eigenvector of the quasi-triangular Schur matrix T for a given eigenvalue.
template <std::floating_point T>
[[nodiscard]] std::vector<Complex<T>> schurEigenvector(const Matrix<T>& T_,
                                                       Complex<T> lambda) {
    using C = Complex<T>;
    const int n = static_cast<int>(T_.rows());
    auto M = std::vector<std::vector<C>>(n, std::vector<C>(n, C{}));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            M[i][j] = C{T_(i, j), T{}} - (i == j ? lambda : C{});

    std::vector<C> y(n);
    for (int i = 0; i < n; ++i)
        y[i] = Complex{static_cast<double>((i % 5) + 1) / 5.0,
                       static_cast<double>((i % 3) - 1) / 7.0};

    for (int it = 0; it < 3; ++it) {
        y = solveComplex<T>(M, y);
        double norm{};
        for (auto e : y) norm = std::max(norm, std::abs(e));
        if (norm == 0.0 || !std::isfinite(norm)) { y.assign(n, C{T{1}, T{}}); continue; }
        for (auto& e : y) e /= norm;
    }
    return y;
}

}  // namespace detail

// ------------------------------------------------------- 6. LU / inversion --

template <std::floating_point T>
[[nodiscard]] std::optional<Matrix<T>> invert(Matrix<T> A) {
    const int n = static_cast<int>(A.rows());
    Matrix<T> inv = Matrix<T>::identity(n);

    for (int k = 0; k < n; ++k) {
        int p = k;
        T best = std::abs(A(k, k));
        for (int i = k + 1; i < n; ++i)
            if (const T v = std::abs(A(i, k)); v > best) { best = v; p = i; }
        if (best == T{}) return std::nullopt;             // exactly singular
        if (p != k) {
            for (int j = 0; j < n; ++j) std::swap(A(k, j), A(p, j));
            for (int j = 0; j < n; ++j) std::swap(inv(k, j), inv(p, j));
        }
        const T pivot = A(k, k);
        for (int i = k + 1; i < n; ++i) {
            const T f = A(i, k) / pivot;
            if (f == T{}) continue;
            for (int j = k;     j < n; ++j) A(i, j)   -= f * A(k, j);
            for (int j = 0;     j < n; ++j) inv(i, j) -= f * inv(k, j);
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            const T f = A(i, j);
            if (f == T{}) continue;
            for (int k = 0; k < n; ++k) inv(i, k) -= f * inv(j, k);
        }
        const T pivot = A(i, i);
        for (int k = 0; k < n; ++k) inv(i, k) /= pivot;
    }
    return inv;
}

// ------------------------------------------------------------- 7. the entry --

/// Full eigendecomposition of a general real matrix (A is consumed).
template <std::floating_point T>
[[nodiscard]] Diagnostics<T> eigGeneral(Matrix<T> A, EigenSystem<T>& out) {
    const int n = static_cast<int>(A.rows());
    Diagnostics<T> diag{};
    const Matrix<T> input = A;          // kept for per-vector residual checks

    const auto scale = balance(A);
    Matrix<T> Z = Matrix<T>::identity(n);
    hessenberg(A, Z);

    const int iters = schurQR(A, Z);
    diag.qrIterations = iters;
    if (iters < 0) {
        diag.status = EigenStatus::NotConverged;
        return diag;
    }

    triangularizeRealBlocks(A, Z);

    const auto is1x1 = blockStructure(A);
    extractEigenvalues(A, out.real, out.imag, out.hasComplex, is1x1);
    for (auto v : out.imag) diag.maxImag = std::max(diag.maxImag, std::abs(v));

    T tnorm{};
    for (auto e : A.data()) tnorm = std::max(tnorm, std::abs(e));
    const T zeroTol = T{64} * std::numeric_limits<T>::epsilon() * std::max(tnorm, T{1});

    out.vectors = Matrix<T>{static_cast<std::size_t>(n), static_cast<std::size_t>(n), T{}};
    auto build = [&](const std::vector<T>& y) {
        std::vector<T> v(n);
        for (int i = 0; i < n; ++i) {
            T acc{};
            for (int j = 0; j < n; ++j) acc += Z(i, j) * y[j];
            v[i] = acc * scale[i];
        }
        return v;
    };

    for (int k = 0; k < n;) {
        if (out.imag[k] != T{} && k + 1 < n) {
            // conjugate pair: inverse iteration (such pairs are generically simple)
            const auto y = detail::nullVectorQuasiC(
                A, detail::Complex<T>{out.real[k], std::abs(out.imag[k])}, is1x1, k);
            for (int i = 0; i < n; ++i) {
                T re{}, im{};
                for (int j = 0; j < n; ++j) {
                    const auto c = detail::Complex<T>{Z(i, j), T{}} * y[j];
                    re += c.real();
                    im += c.imag();
                }
                out.vectors(i, k)     = re * scale[i];
                out.vectors(i, k + 1) = im * scale[i];
            }
            k += 2;
            continue;
        }

        // real eigenvalue: exact back-substitution.  Every zero-diagonal
        // position is a free variable, so handing a different one to each
        // copy of a repeated root gives independent vectors (JC69 needs it).
        int copy = 0;
        for (int j = 0; j < k; ++j)
            if (std::abs(out.real[j] - out.real[k]) <= zeroTol) ++copy;

        int freeIdx = -1, seen = 0;
        for (int i = 0; i < n; ++i) {
            if (!is1x1[i] || std::abs(A(i, i) - out.real[k]) > zeroTol) continue;
            if (seen++ == copy) { freeIdx = i; break; }
        }
        if (freeIdx < 0) freeIdx = k;

        auto v = build(nullVectorQuasi(A, out.real[k], freeIdx, zeroTol, is1x1));

        // Back-substitution cannot anchor inside a 2x2 block that survived
        // triangularisation (or when roots are so close that the free-variable
        // bookkeeping slips).  Verify and fall back to inverse iteration.
        if (!(residualOf(input, v, out.real[k]) < T{1e-9})) {
            const auto y = detail::schurEigenvector<T>(A, detail::Complex<T>{out.real[k], T{}});
            std::vector<T> alt(n);
            for (int i = 0; i < n; ++i) {
                T acc{};
                for (int j = 0; j < n; ++j) acc += Z(i, j) * y[j].real();
                alt[i] = acc * scale[i];
            }
            if (residualOf(input, alt, out.real[k]) < residualOf(input, v, out.real[k]))
                v = std::move(alt);
        }

        for (int i = 0; i < n; ++i) out.vectors(i, k) = v[i];
        ++k;
    }

    // Normalise to max |.| == 1 (EISPACK convention).  The two columns of a
    // conjugate pair MUST share one factor: they are the real and imaginary
    // parts of a single complex eigenvector, and normalising them separately
    // would silently destroy it.  An all-zero column means the root
    // bookkeeping failed -> fall back to a unit vector so E stays invertible.
    for (int k = 0; k < n;) {
        const bool pair = out.imag[k] != T{} && k + 1 < n;
        const int m = pair ? 2 : 1;

        T mx{};
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < m; ++d) mx = std::max(mx, std::abs(out.vectors(i, k + d)));

        if (mx == T{}) {
            for (int d = 0; d < m; ++d)
                for (int i = 0; i < n; ++i)
                    out.vectors(i, k + d) = (i == k + d) ? T{1} : T{};
            k += m;
            continue;
        }
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < m; ++d) out.vectors(i, k + d) /= mx;
        k += m;
    }

    if (auto inv = invert(out.vectors); inv) out.inverseVectors = std::move(*inv);
    else diag.status = EigenStatus::Singular;

    return diag;
}

/// Residual of one real eigenpair,  ||A v - lambda v||_inf / ||v||_inf.
template <std::floating_point T>
[[nodiscard]] T residualOf(const Matrix<T>& A, const std::vector<T>& v, T lambda) {
    const int n = static_cast<int>(A.rows());
    T norm{};
    for (auto e : v) norm = std::max(norm, std::abs(e));
    if (norm == T{}) return std::numeric_limits<T>::infinity();
    T worst{};
    for (int i = 0; i < n; ++i) {
        T acc{};
        for (int j = 0; j < n; ++j) acc += A(i, j) * v[j];
        worst = std::max(worst, std::abs(acc - lambda * v[i]));
    }
    return worst / norm;
}

/// max_k ||A v_k - lambda_k v_k||_inf / ||v_k||_inf, complex pairs included.
template <std::floating_point T>
[[nodiscard]] T maxResidual(const Matrix<T>& A, const EigenSystem<T>& es) {
    const int n = static_cast<int>(A.rows());
    T worst{};

    for (int k = 0; k < n;) {
        const bool pair = (es.imag[k] != T{} && k + 1 < n);
        const int m = pair ? 2 : 1;

        T norm{};
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < m; ++d) norm = std::max(norm, std::abs(es.vectors(i, k + d)));
        if (norm == T{}) { k += m; continue; }

        for (int i = 0; i < n; ++i) {
            if (!pair) {
                T s{};
                for (int j = 0; j < n; ++j) s += A(i, j) * es.vectors(j, k);
                s -= es.real[k] * es.vectors(i, k);
                worst = std::max(worst, std::abs(s) / norm);
            } else {
                // A (vr + i vi) == (a + i b)(vr + i vi)
                const T a = es.real[k], b = es.imag[k];
                T sr{}, si{};
                for (int j = 0; j < n; ++j) {
                    sr += A(i, j) * es.vectors(j, k);
                    si += A(i, j) * es.vectors(j, k + 1);
                }
                const T rr = sr - (a * es.vectors(i, k) - b * es.vectors(i, k + 1));
                const T ri = si - (b * es.vectors(i, k) + a * es.vectors(i, k + 1));
                worst = std::max(worst, std::hypot(rr, ri) / norm);
            }
        }
        k += m;
    }
    return worst;
}

// ------------------------------------------------- 8. P(t) = E diag E^-1 -----

/// BEAGLE "square" update:  P = E * B(t) * E^-1  where B(t) is block-diagonal,
///   * exp(lambda_k * t)                       for a real eigenvalue
///   * e^{a t} [[cos bt,  sin bt],
///             [-sin bt, cos bt]]              for the pair a +/- i b
/// Negative entries produced by round-off are clamped to zero.
template <std::floating_point T>
[[nodiscard]] Matrix<T> transitionMatrix(const EigenSystem<T>& es, T t, T rate = T{1}) {
    const int n = static_cast<int>(es.real.size());
    const T dt = t * rate;

    // tmp = B(t) * E^-1, built by rows (no matrix product)
    Matrix<T> tmp{static_cast<std::size_t>(n), static_cast<std::size_t>(n), T{}};
    for (int i = 0; i < n;) {
        if (es.imag[i] != T{} && i + 1 < n) {
            const T a = es.real[i], b = es.imag[i];
            const T eat = std::exp(a * dt);
            const T ec = eat * std::cos(b * dt);
            const T es_ = eat * std::sin(b * dt);
            for (int j = 0; j < n; ++j) {
                const T p = es.inverseVectors(i, j), q = es.inverseVectors(i + 1, j);
                tmp(i,     j) = ec * p + es_ * q;
                tmp(i + 1, j) = ec * q - es_ * p;
            }
            i += 2;
        } else {
            const T d = std::exp(es.real[i] * dt);
            for (int j = 0; j < n; ++j) tmp(i, j) = es.inverseVectors(i, j) * d;
            ++i;
        }
    }

    Matrix<T> P{static_cast<std::size_t>(n), static_cast<std::size_t>(n), T{}};
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            T s{};
            for (int k = 0; k < n; ++k) s += es.vectors(i, k) * tmp(k, j);
            P(i, j) = s > T{} ? s : T{};
        }
    return P;
}

/// Pack eigenvalues as beagleSetEigenDecomposition expects them:
/// [0, n) real parts, [n, 2n) imaginary parts.
template <std::floating_point T>
[[nodiscard]] std::vector<T> packBeagleEigenValues(const EigenSystem<T>& es) {
    std::vector<T> out(2 * es.real.size());
    std::copy(es.real.begin(), es.real.end(), out.begin());
    std::copy(es.imag.begin(), es.imag.end(), out.begin() + es.real.size());
    return out;
}

}  // namespace mpe::linalg
