#ifndef PYML_CORE_PRIORS_HPP
#define PYML_CORE_PRIORS_HPP

/* ===========================================================================
 *  core/priors.hpp — the priors a MrBayes-style DNA run needs
 *
 *  Matches the showmodel output of a default MrBayes DNA analysis:
 *      Pi   ~ Dirichlet(1,1,1,1)
 *      Tau  ~ all topologies equally probable
 *      V    ~ GammaDir(1.0, 0.1, 1.0, 1.0)
 *  The partition rate multiplier is omitted: there is one partition.
 * =========================================================================== */

#include "rng.hpp"
#include "state.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyml::core {

class Prior {
public:
    virtual ~Prior() = default;
    /// Diagnostic only, and deliberately not overridable from Python -- see
    /// the note on Move::Name().
    [[nodiscard]] virtual std::string Name() const {
        return param.empty() ? "Prior" : "Prior(" + param + ")";
    }
    /// Log density at the current state.  May return -inf.
    [[nodiscard]] virtual double LogPrior(const Core& core) const = 0;
    std::string param;
};

/// Constant: contributes nothing.  Used for Tau, where every topology has the
/// same prior probability, and for parameters that are fixed.
class FlatPrior : public Prior {
public:
    explicit FlatPrior(std::string p = "Tau") { param = std::move(p); }
    [[nodiscard]] std::string Name() const override { return "Flat(" + param + ")"; }
    [[nodiscard]] double LogPrior(const Core&) const override { return 0.0; }
};

/// Dirichlet(a) on the stationary frequencies.
class DirichletPrior : public Prior {
public:
    explicit DirichletPrior(std::vector<double> alpha, std::string p = "Pi")
        : alpha_(std::move(alpha)) {
        param = std::move(p);
        if (alpha_.empty()) throw std::invalid_argument("DirichletPrior: empty alpha");
        for (double a : alpha_)
            if (a <= 0.0) throw std::invalid_argument("DirichletPrior: alpha must be positive");
    }

    [[nodiscard]] std::string Name() const override { return "Dirichlet(" + param + ")"; }

    [[nodiscard]] double LogPrior(const Core& core) const override {
        const auto& pi = core.freqs();
        if (pi.size() != alpha_.size())
            throw std::invalid_argument("DirichletPrior: alpha does not match the state count");
        double acc = -LogDirichletConstant(alpha_.data(), static_cast<int>(alpha_.size()));
        for (std::size_t i = 0; i < pi.size(); ++i) {
            // The floor is part of the target: frequency moves reject below it
            // too, and the two have to agree or the chain samples something
            // other than this density.  See kMinFreq in state.hpp.
            if (pi[i] < kMinFreq) return -std::numeric_limits<double>::infinity();
            acc += (alpha_[i] - 1.0) * std::log(pi[i]);
        }
        return acc;
    }

    const std::vector<double>& alpha() const { return alpha_; }

private:
    std::vector<double> alpha_;
};

/// Compound Dirichlet-gamma on the branch lengths:
///     L = sum of branch lengths      ~ Gamma(shape, rate)
///     p_i = v_i / L                  ~ Dirichlet(alpha, ..., alpha)
///
/// The density is written over the branch lengths themselves, so the change of
/// variables v = L*p contributes the Jacobian L^(k-1):
///     log f(v) = log f_Gamma(L) + log f_Dir(p) - (k-1) log L
/// Getting that term wrong is a common source of a subtly wrong posterior, so
/// it is spelled out rather than folded into the Dirichlet constant.
class GammaDirPrior : public Prior {
public:
    GammaDirPrior(double shape = 1.0, double rate = 0.1, double alpha = 1.0,
                  std::string p = "V")
        : shape_(shape), rate_(rate), alpha_(alpha) {
        param = std::move(p);
        if (shape_ <= 0.0 || rate_ <= 0.0 || alpha_ <= 0.0)
            throw std::invalid_argument("GammaDirPrior: parameters must be positive");
    }

    [[nodiscard]] std::string Name() const override { return "GammaDir(" + param + ")"; }

    [[nodiscard]] double LogPrior(const Core& core) const override {
        const Tree& t = core.tree();
        const int root = Root(t);
        const int k = BranchCount(t);
        if (k < 1) return 0.0;

        double length = 0.0;
        for (int v = 0; v < t.NODE_COUNT; ++v) {
            if (v == root) continue;
            const double bl = At(t, v).branch_length;
            if (bl < 0.0) return -std::numeric_limits<double>::infinity();
            length += bl;
        }
        if (!(length > 0.0)) return -std::numeric_limits<double>::infinity();

        // log Gamma(L; shape, rate) with rate = 1/scale
        const double log_gamma = shape_ * std::log(rate_) - std::lgamma(shape_) +
                                 (shape_ - 1.0) * std::log(length) - rate_ * length;

        // Dirichlet(alpha,...,alpha) on the proportions
        const double log_const = std::lgamma(alpha_ * static_cast<double>(k)) -
                                 static_cast<double>(k) * std::lgamma(alpha_);
        double log_dir = -log_const;
        for (int v = 0; v < t.NODE_COUNT; ++v) {
            if (v == root) continue;
            const double p = At(t, v).branch_length / length;
            if (!(p > 0.0)) return -std::numeric_limits<double>::infinity();
            log_dir += (alpha_ - 1.0) * std::log(p);
        }

        // Jacobian of v = L * p
        return log_gamma + log_dir - static_cast<double>(k - 1) * std::log(length);
    }

    double shape() const { return shape_; }
    double rate() const { return rate_; }
    double alpha() const { return alpha_; }

private:
    double shape_, rate_, alpha_;
};

}  // namespace pyml::core

#endif  // PYML_CORE_PRIORS_HPP
