#ifndef PYML_CORE_RNG_HPP
#define PYML_CORE_RNG_HPP

/* Deterministic RNG for the MCMC core.  Seeded from Python so a run can be
 * reproduced exactly, which is what makes the move-by-move tests meaningful. */

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace pyml::core {

class Rng {
public:
    explicit Rng(std::uint64_t seed) : gen_(seed) {}

    /// Uniform on [0,1).
    [[nodiscard]] double Uniform() { return uni_(gen_); }

    /// Uniform integer on [0, n).  n must be positive.
    [[nodiscard]] int Below(int n) {
        if (n <= 0) throw std::invalid_argument("Rng::Below needs n > 0");
        return static_cast<int>(uni_(gen_) * static_cast<double>(n));
    }

    /// Uniform on [lo, hi].
    [[nodiscard]] double Uniform(double lo, double hi) {
        return lo + (hi - lo) * Uniform();
    }

    [[nodiscard]] double Normal(double mean, double sd) {
        std::normal_distribution<double> d(mean, sd);
        return d(gen_);
    }

    /// Gamma(shape, scale).  shape must be positive.
    [[nodiscard]] double Gamma(double shape, double scale = 1.0) {
        if (shape <= 0.0) throw std::invalid_argument("Rng::Gamma needs shape > 0");
        std::gamma_distribution<double> d(shape, scale);
        return d(gen_);
    }

    [[nodiscard]] double Exponential(double rate) {
        if (rate <= 0.0) throw std::invalid_argument("Rng::Exponential needs rate > 0");
        std::exponential_distribution<double> d(rate);
        return d(gen_);
    }

    /// Symmetric multiplier in log space: exp(lambda * (u - 1/2)).
    /// Returns both the multiplier and its log, since every move that uses one
    /// needs the log for the Hastings ratio.
    [[nodiscard]] double LogMultiplier(double lambda) {
        return lambda * (Uniform() - 0.5);
    }

private:
    std::mt19937_64 gen_;
    std::uniform_real_distribution<double> uni_{0.0, 1.0};
};

/// Draw a Dirichlet(alpha) vector using independent Gamma variates.
/// Out-of-range or non-positive concentrations throw.
inline void DirichletSample(Rng& rng, const double* alpha, int n, double* out) {
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        if (alpha[i] <= 0.0) throw std::invalid_argument("Dirichlet: alpha must be positive");
        out[i] = rng.Gamma(alpha[i], 1.0);
        total += out[i];
    }
    if (!(total > 0.0)) throw std::runtime_error("Dirichlet: degenerate draw");
    for (int i = 0; i < n; ++i) out[i] /= total;
}

/// log of the Dirichlet normalising constant:
///   log Gamma(sum a) - sum log Gamma(a)
[[nodiscard]] inline double LogDirichletConstant(const double* alpha, int n) {
    double sum = 0.0, acc = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += alpha[i];
        acc += std::lgamma(alpha[i]);
    }
    return std::lgamma(sum) - acc;
}

}  // namespace pyml::core

#endif  // PYML_CORE_RNG_HPP
