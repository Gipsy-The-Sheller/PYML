#ifndef PYML_CORE_MOVES_HPP
#define PYML_CORE_MOVES_HPP

/* ===========================================================================
 *  core/moves.hpp — Metropolis proposals
 *
 *  Each move mutates the state and returns the log Hastings ratio
 *  log[q(x|x') / q(x'|x)], which the driver adds to the log posterior
 *  difference before the accept/reject test:
 *
 *      accept if  log(u) < logPost(x') - logPost(x) + logHR,   u ~ U(0,1)
 *
 *  Two of the scaling exponents below are marked as "determined by test".
 *  Multiplier-style moves on a k-dimensional parameter pick up a power of the
 *  multiplier, and the literature writes that power inconsistently because it
 *  depends on how the proposal density is normalised.  Rather than trust a
 *  derivation, each exponent is a constructor argument with a default, and
 *  test_priors.cpp runs the chain with a constant likelihood and checks that
 *  the samples reproduce the prior -- which pins the exponent down exactly.
 * =========================================================================== */

#include "rng.hpp"
#include "state.hpp"

#include <algorithm>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyml::core {

/// Returned by Propose when the proposal came out degenerate and the state was
/// deliberately left untouched.  The driver then always rejects, since
/// log(u) < -inf is never true.  A move must not both mutate and return this.
inline constexpr double kReject = -std::numeric_limits<double>::infinity();

class Move {
public:
    virtual ~Move() = default;

    /// Purely for diagnostics.  Not overridable from Python: a subclass there
    /// would want to expose it as a plain attribute named `name`, and the
    /// override machinery can only call something callable.  `param` carries
    /// the useful label anyway.
    [[nodiscard]] virtual std::string Name() const {
        return param.empty() ? "Move" : "Move(" + param + ")";
    }

    /// Mutate the state and return the log Hastings ratio.  Must leave the
    /// state declared-valid (tree links consistent, frequencies summing to 1).
    virtual double Propose(Core& core, Rng& rng) = 0;

    /// Relative proposal frequency.  Zero disables the move.
    double weight = 1.0;
    std::string param;          // MrBayes-style label, e.g. "V" or "Pi"

    long n_proposed = 0;
    long n_accepted = 0;

    /// Acceptance rate this move is tuned towards, MrBayes' Targetrate.
    /// Zero disables tuning (NNI has nothing to scale anyway).
    double target_rate = 0.0;

    /// Bookkeeping for the tuner: totals as of the last adjustment.
    long tune_mark = 0;
    long accept_mark = 0;

    [[nodiscard]] double AcceptanceRate() const {
        return n_proposed ? static_cast<double>(n_accepted) / static_cast<double>(n_proposed) : 0.0;
    }
    [[nodiscard]] double WindowRate() const {
        const long n = n_proposed - tune_mark;
        return n > 0 ? static_cast<double>(n_accepted - accept_mark) / static_cast<double>(n) : 0.0;
    }

    /// Scale the tuning parameter.  `factor` > 1 means "propose bigger
    /// steps", < 1 means smaller.  Moves whose parameter runs the other way
    /// (a Dirichlet concentration: larger means smaller steps) invert it.
    virtual void ScaleTuning(double factor) { (void)factor; }

    /// Human-readable tuning parameter, for the trace header.
    [[nodiscard]] virtual double tuning() const { return 0.0; }
};

/* ------------------------------------------------ branch length moves --- */

/// Multiply one randomly chosen branch length by exp(lambda * (u - 1/2)).
///
/// One-dimensional, so the density ratio is exact.  With s = log m drawn from a
/// density symmetric about zero and x' = e^s x:
///     q(x -> x') = g(s) / x'      (the Jacobian of s -> x' is 1/x')
///     q(x' -> x) = g(-s) / x
/// so the log Hastings ratio is +s = +log m.  Getting the sign wrong is not a
/// subtle error: it makes every up-scaling carry a penalty, and the chain
/// collapses onto zero branch lengths.  test_priors.cpp catches it.
class Multiplier : public Move {
public:
    explicit Multiplier(double lambda = 1.386, std::string p = "V")
        : lambda_(lambda) {
        param = std::move(p);
        target_rate = 0.25;                    // MrBayes' Targetrate
    }

    [[nodiscard]] std::string Name() const override { return "Multiplier(" + param + ")"; }

    double Propose(Core& core, Rng& rng) override {
        Tree& t = core.tree();
        const int root = Root(t);
        const int movable = t.NODE_COUNT - 1;          // every node but the root
        if (movable < 1) return 0.0;
        int node = rng.Below(movable);
        if (node >= root) ++node;                      // skip the root

        const double logm = rng.LogMultiplier(lambda_);
        auto& bl = At(t, node).branch_length;
        if (!(bl > 0.0)) return kReject;
        const double scaled = bl * std::exp(logm);
        if (!(scaled > 0.0) || !std::isfinite(scaled)) return kReject;
        bl = scaled;
        return logm;
    }

    double lambda() const { return lambda_; }
    void set_lambda(double v) { lambda_ = v; }

    void ScaleTuning(double factor) override {
        lambda_ = std::clamp(lambda_ * factor, 1e-4, 100.0);
    }
    [[nodiscard]] double tuning() const override { return lambda_; }

private:
    double lambda_;
};

/// Multiply every branch length by the same multiplier: a move along the
/// tree-length axis.  The posterior ratio carries no other information about
/// which ray we are on, so the exponent is the only thing that fixes the
/// stationary distribution -- see the header note.
class TLMultiplier : public Move {
public:
    explicit TLMultiplier(double lambda = 1.386, std::string p = "V",
                          int exponent = -1)
        : lambda_(lambda), exponent_(exponent) {
        param = std::move(p);
        target_rate = 0.25;
    }

    [[nodiscard]] std::string Name() const override { return "TLMultiplier(" + param + ")"; }

    double Propose(Core& core, Rng& rng) override {
        Tree& t = core.tree();
        const int root = Root(t);
        const int k = BranchCount(t);
        if (k < 1) return 0.0;
        const double logm = rng.LogMultiplier(lambda_);
        const double m = std::exp(logm);
        if (!(m > 0.0) || !std::isfinite(m)) return kReject;
        for (int v = 0; v < t.NODE_COUNT; ++v) {
            if (v == root) continue;
            const double scaled = At(t, v).branch_length * m;
            if (!(scaled > 0.0) || !std::isfinite(scaled)) return kReject;
            At(t, v).branch_length = scaled;
        }
        const int e = exponent_ < 0 ? k : exponent_;
        return static_cast<double>(e) * logm;
    }

    double lambda() const { return lambda_; }
    void set_lambda(double v) { lambda_ = v; }
    int exponent() const { return exponent_; }
    void set_exponent(int e) { exponent_ = e; }

    void ScaleTuning(double factor) override {
        lambda_ = std::clamp(lambda_ * factor, 1e-4, 100.0);
    }
    [[nodiscard]] double tuning() const override { return lambda_; }

private:
    double lambda_;
    int exponent_;      // -1 means "use the branch count"
};

/// Nearest-neighbour interchange on a randomly chosen internal edge.
///
/// Every unrooted binary tree over N tips has the same N-3 internal edges, so
/// choosing one uniformly and then one of its two alternatives uniformly is a
/// symmetric proposal: log Hastings ratio exactly zero.
class NNI : public Move {
public:
    explicit NNI(std::string p = "Tau,V") { param = std::move(p); }

    [[nodiscard]] std::string Name() const override { return "NNI(" + param + ")"; }

    double Propose(Core& core, Rng& rng) override {
        Tree& t = core.tree();
        const auto edges = InternalEdges(t);
        if (edges.empty()) return 0.0;                 // N == 3: no internal edge

        const int v = edges[static_cast<std::size_t>(rng.Below(static_cast<int>(edges.size())))];
        const int u = At(t, v).parent_idx;             // u is internal too

        // Split each side into "the two subtrees that are not on this edge".
        // v's side: its two children.  u's side: u's other children, and if u
        // is the root it has two of them.
        const int v0 = At(t, v).child_idx[0];
        const int v1 = At(t, v).child_idx[1];
        std::vector<int> u_side;
        for (int s = 0; s < At(t, u).n_child; ++s) {
            const int c = At(t, u).child_idx[static_cast<std::size_t>(s)];
            if (c != v) u_side.push_back(c);
        }
        if (u_side.empty()) return 0.0;

        // Two alternatives: swap v0 with either member of u_side, or swap v1
        // with either.  Choosing uniformly among the (child, neighbour) pairs
        // reaches each of the two distinct topologies with equal probability.
        const int pick_child = rng.Below(2);
        const int pick_side = rng.Below(static_cast<int>(u_side.size()));
        const int a = pick_child == 0 ? v0 : v1;
        const int b = u_side[static_cast<std::size_t>(pick_side)];

        // Detach both before attaching either.  When u is the root it already
        // has three children, so attaching first would momentarily ask for a
        // fourth and AppendChild would refuse.
        DetachChild(t, v, a);
        DetachChild(t, u, b);
        AppendChild(t, u, a);
        AppendChild(t, v, b);
        return 0.0;
    }
};

/* -------------------------------------------------- frequency moves --- */

/// Propose new stationary frequencies from Dirichlet(alpha * pi).
class DirichletFreqs : public Move {
public:
    explicit DirichletFreqs(double alpha = 100.0, std::string p = "Pi")
        : alpha_(alpha) {
        param = std::move(p);
        target_rate = 0.25;
    }

    [[nodiscard]] std::string Name() const override { return "Dirichlet(" + param + ")"; }

    double Propose(Core& core, Rng& rng) override {
        const int n = core.n_states();
        const std::vector<double>& pi = core.freqs();
        std::vector<double> a(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) a[static_cast<std::size_t>(i)] = alpha_ * pi[static_cast<std::size_t>(i)];

        std::vector<double> prop(static_cast<std::size_t>(n));
        DirichletSample(rng, a.data(), n, prop.data());

        // A Gamma draw can underflow to exactly zero, and a frequency that
        // small makes Q's eigen decomposition fail.  Bail before touching the
        // state; the prior would reject it anyway.
        for (int i = 0; i < n; ++i)
            if (!(prop[static_cast<std::size_t>(i)] >= kMinFreq)) return kReject;

        // log q(pi -> pi') - log q(pi' -> pi), both Dirichlet densities with
        // the concentration scaled by the *other* point.
        std::vector<double> a_rev(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) a_rev[static_cast<std::size_t>(i)] = alpha_ * prop[static_cast<std::size_t>(i)];

        const double logq_fwd = LogDirichletConstant(a.data(), n) +
            LogDirichletDensity(a.data(), prop.data(), n);
        const double logq_rev = LogDirichletConstant(a_rev.data(), n) +
            LogDirichletDensity(a_rev.data(), pi.data(), n);

        if (!core.TrySetFreqs(std::move(prop))) return kReject;
        return logq_rev - logq_fwd;
    }

    double alpha() const { return alpha_; }
    void set_alpha(double v) { alpha_ = v; }

    /// Inverted on purpose: a *higher* Dirichlet concentration puts the draw
    /// closer to the current point, so bigger steps mean a smaller alpha.
    void ScaleTuning(double factor) override {
        alpha_ = std::clamp(alpha_ / factor, 1e-2, 1e6);
    }
    [[nodiscard]] double tuning() const override { return alpha_; }

private:
    [[nodiscard]] static double LogDirichletDensity(const double* a, const double* x, int n) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) {
            if (x[i] <= 0.0) return -std::numeric_limits<double>::infinity();
            acc += (a[i] - 1.0) * std::log(x[i]);
        }
        return acc;
    }

    double alpha_;
};

/// Slide two frequencies apart by delta, with reflection at the boundary so
/// the proposal stays on the simplex and remains symmetric (log HR = 0).
class SliderFreqs : public Move {
public:
    explicit SliderFreqs(double delta = 0.2, std::string p = "Pi")
        : delta_(delta) {
        param = std::move(p);
        target_rate = 0.25;
    }

    [[nodiscard]] std::string Name() const override { return "Slider(" + param + ")"; }

    double Propose(Core& core, Rng& rng) override {
        const int n = core.n_states();
        if (n < 2) return 0.0;
        std::vector<double> pi = core.freqs();

        const int i = rng.Below(n);
        int j = rng.Below(n - 1);
        if (j >= i) ++j;                                // j != i, uniform

        const double step = delta_ * (2.0 * rng.Uniform() - 1.0);
        double a = pi[static_cast<std::size_t>(i)] + step;
        double b = pi[static_cast<std::size_t>(j)] - step;

        // Reflect into (0,1) preserving the pair's sum, which keeps the map
        // an involution and so the proposal symmetric.
        const double sum = a + b;
        a = std::fmod(a, 2.0 * sum);
        if (a < 0.0) a += 2.0 * sum;
        if (a > sum) a = 2.0 * sum - a;
        b = sum - a;
        // Reject rather than accept-with-ratio-zero.  Returning 0 without
        // mutating would make the driver accept a no-op every time, which
        // leaves the target alone but makes the acceptance rate meaningless.
        if (!(a >= kMinFreq) || !(b >= kMinFreq)) return kReject;

        pi[static_cast<std::size_t>(i)] = a;
        pi[static_cast<std::size_t>(j)] = b;
        if (!core.TrySetFreqs(std::move(pi))) return kReject;
        return 0.0;
    }

    double delta() const { return delta_; }
    void set_delta(double v) { delta_ = v; }

    void ScaleTuning(double factor) override {
        delta_ = std::clamp(delta_ * factor, 1e-5, 1.0);
    }
    [[nodiscard]] double tuning() const override { return delta_; }

private:
    double delta_;
};

}  // namespace pyml::core

#endif  // PYML_CORE_MOVES_HPP
