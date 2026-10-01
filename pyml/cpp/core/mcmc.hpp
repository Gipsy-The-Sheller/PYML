#ifndef PYML_CORE_MCMC_HPP
#define PYML_CORE_MCMC_HPP

/* ===========================================================================
 *  core/mcmc.hpp — the Metropolis-Hastings driver
 *
 *  One iteration:
 *      pick a move by weight
 *      snapshot the state
 *      logHR = move->Propose(...)
 *      accept if  log u  <  (logpost' - logpost) + logHR
 *      otherwise restore the snapshot
 *
 *  Only one likelihood evaluation per iteration: on rejection the state goes
 *  back exactly, so the standing log posterior is still the value at the
 *  current point and there is nothing to recompute.
 *
 *  `sample_likelihood = false` drops the likelihood from the target, which
 *  turns the chain into a sampler of the prior.  That is not a toy mode: it is
 *  how the Hastings ratios are checked.  With the likelihood gone the
 *  stationary distribution is known in closed form, so any wrong ratio shows
 *  up as a wrong sampled distribution rather than as a subtle bias.
 * =========================================================================== */

#include "moves.hpp"
#include "priors.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyml::core {

class Mcmc {
public:
    Mcmc(Core& core, std::vector<std::shared_ptr<Move>> moves,
         std::vector<std::shared_ptr<Prior>> priors, std::uint64_t seed = 1)
        : core_(core), moves_(std::move(moves)), priors_(std::move(priors)), rng_(seed) {
        if (moves_.empty()) throw std::invalid_argument("Mcmc: no moves");
        for (const auto& m : moves_)
            if (m->weight < 0.0) throw std::invalid_argument("Mcmc: negative move weight");
        Refresh();
    }

    /// When false the likelihood is left out of the target, so the chain
    /// samples the prior.  See the header note.
    bool sample_likelihood = true;

    /// Tune every move with a target rate, in proportion to its own volume.
    bool auto_tune = true;

    /// Proposals a move makes before its rate is compared with the target.
    long tune_window = 200;

    /// Turns on trace recording.  `interval` is in iterations; the trace is
    /// thinned by that factor.  `with_trees` also stores a Newick string per
    /// sample, which is what a MrBayes .t file holds.
    void EnableTrace(long interval = 100, bool with_trees = false,
                     bool with_branches = false) {
        if (interval < 1) throw std::invalid_argument("trace interval must be positive");
        trace_interval_ = interval;
        trace_trees_ = with_trees;
        trace_branches_ = with_branches;
        trace_iter_.clear();
        trace_loglik_.clear();
        trace_logprior_.clear();
        trace_logpost_.clear();
        trace_treelen_.clear();
        trace_freqs_.clear();
        trace_brlen_.clear();
        trace_newick_.clear();
        trace_next_ = 0;
    }

    /// Drop the first `n` recorded samples: burn-in.
    void DiscardTrace(long n) {
        const auto cut = static_cast<std::size_t>(std::max<long>(0, n));
        const auto clip = [cut](auto& v) {
            if (cut >= v.size()) v.clear();
            else v.erase(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(cut));
        };
        clip(trace_iter_);
        clip(trace_loglik_);
        clip(trace_logprior_);
        clip(trace_logpost_);
        clip(trace_treelen_);
        clip(trace_freqs_);
        clip(trace_brlen_);
        clip(trace_newick_);
    }

    [[nodiscard]] const std::vector<long>& trace_iteration() const noexcept { return trace_iter_; }
    [[nodiscard]] const std::vector<double>& trace_loglik() const noexcept { return trace_loglik_; }
    [[nodiscard]] const std::vector<double>& trace_logprior() const noexcept { return trace_logprior_; }
    [[nodiscard]] const std::vector<double>& trace_logposterior() const noexcept { return trace_logpost_; }
    [[nodiscard]] const std::vector<double>& trace_tree_length() const noexcept { return trace_treelen_; }
    [[nodiscard]] const std::vector<std::vector<double>>& trace_freqs() const noexcept { return trace_freqs_; }
    [[nodiscard]] const std::vector<std::vector<double>>& trace_branch_lengths() const noexcept { return trace_brlen_; }
    [[nodiscard]] const std::vector<std::string>& trace_newick() const noexcept { return trace_newick_; }
    [[nodiscard]] bool tracing() const noexcept { return trace_interval_ > 0; }

    /// One proposal.
    void Step() {
        ++iteration_;
        Move* move = PickMove();
        if (move == nullptr) return;

        snap_.Capture(core_.tree(), core_.freqs());
        const double log_hr = move->Propose(core_, rng_);
        ++move->n_proposed;

        double log_lik = 0.0;
        const double proposed = LogTarget(&log_lik);
        const double log_alpha = proposed - log_post_ + log_hr;

        if (std::log(rng_.Uniform()) < log_alpha) {
            log_post_ = proposed;
            log_prior_ = proposed - log_lik;
            if (sample_likelihood) log_lik_ = log_lik;
            ++move->n_accepted;
        } else {
            core_.SetFreqs(snap_.Restore(core_.tree()));
        }

        if (auto_tune) Tune();
        if (trace_interval_ > 0 && iteration_ >= trace_next_) {
            Record();
            trace_next_ = iteration_ + trace_interval_;
        }
    }

    void Run(long iterations) {
        if (iterations < 0) throw std::invalid_argument("Mcmc::Run needs a non-negative count");
        for (long i = 0; i < iterations; ++i) Step();
    }

    [[nodiscard]] long iterations() const noexcept { return iteration_; }

    [[nodiscard]] double log_likelihood() const noexcept { return log_lik_; }
    [[nodiscard]] double log_prior() const noexcept { return log_prior_; }
    [[nodiscard]] double log_posterior() const noexcept { return log_post_; }
    [[nodiscard]] double tree_length() const { return core_.TreeLength(); }
    [[nodiscard]] const std::vector<double>& freqs() const noexcept { return core_.freqs(); }
    [[nodiscard]] Core& core() noexcept { return core_; }
    [[nodiscard]] const std::vector<std::shared_ptr<Move>>& moves() const noexcept {
        return moves_;
    }
    [[nodiscard]] const std::vector<std::shared_ptr<Prior>>& priors() const noexcept {
        return priors_;
    }

    /// Recompute the standing log posterior from scratch.  Call after mutating
    /// the state from outside.
    void Refresh() {
        double log_lik = 0.0;
        log_post_ = LogTarget(&log_lik);
        log_prior_ = log_post_ - log_lik;
        if (sample_likelihood) log_lik_ = log_lik;
    }

private:
    [[nodiscard]] double LogTarget(double* log_lik_out) const {
        double lp = 0.0;
        for (const auto& p : priors_) {
            lp += p->LogPrior(core_);
            // A state the prior rules out is going to be rejected whatever the
            // likelihood says, so do not spend a pruning pass on it.
            if (lp == -std::numeric_limits<double>::infinity()) {
                if (log_lik_out) *log_lik_out = 0.0;
                return lp;
            }
        }
        const double ll = sample_likelihood ? const_cast<Core&>(core_).LogLik() : 0.0;
        if (log_lik_out) *log_lik_out = ll;
        return ll + lp;
    }

    /// MrBayes-style tuning: every `tune_window` proposals of a move, compare
    /// its acceptance rate with its target and scale the tuning parameter by a
    /// fixed factor.  Iterated, this walks the rate towards the target without
    /// needing an initial guess, which is the whole point -- the hand-picked
    /// defaults left the frequency moves accepting about one proposal in a
    /// hundred, near enough frozen to matter.
    void Tune() {
        for (const auto& m : moves_) {
            if (m->target_rate <= 0.0) continue;
            const long n = m->n_proposed - m->tune_mark;
            if (n < tune_window) continue;
            const double rate = static_cast<double>(m->n_accepted - m->accept_mark) /
                                static_cast<double>(n);
            const double factor = rate > m->target_rate ? 1.2 : (1.0 / 1.2);
            m->ScaleTuning(factor);
            m->tune_mark = m->n_proposed;
            m->accept_mark = m->n_accepted;
        }
    }

    void Record() {
        trace_iter_.push_back(iteration_);
        trace_loglik_.push_back(log_lik_);
        trace_logprior_.push_back(log_prior_);
        trace_logpost_.push_back(log_post_);
        trace_treelen_.push_back(core_.TreeLength());
        trace_freqs_.push_back(core_.freqs());
        // Columns follow BranchLengthsCanonical, so a column is the same
        // physical branch for every sample even though NNI keeps relinking.
        if (trace_branches_) trace_brlen_.push_back(BranchLengthsCanonical(core_.tree()));
        if (trace_trees_) trace_newick_.push_back(ToNewick(core_.tree(), core_.data().tip_name));
    }

    [[nodiscard]] Move* PickMove() {
        double total = 0.0;
        for (const auto& m : moves_) total += m->weight;
        if (!(total > 0.0)) return nullptr;
        double u = rng_.Uniform() * total;
        for (const auto& m : moves_) {
            u -= m->weight;
            if (u < 0.0) return m.get();
        }
        return moves_.back().get();
    }

    Core& core_;
    std::vector<std::shared_ptr<Move>> moves_;
    std::vector<std::shared_ptr<Prior>> priors_;
    Rng rng_;
    Snapshot snap_;
    double log_post_ = 0.0, log_prior_ = 0.0, log_lik_ = 0.0;
    long iteration_ = 0;

    long trace_interval_ = 0;
    long trace_next_ = 0;
    bool trace_trees_ = false;
    bool trace_branches_ = false;
    std::vector<long> trace_iter_;
    std::vector<double> trace_loglik_, trace_logprior_, trace_logpost_, trace_treelen_;
    std::vector<std::vector<double>> trace_freqs_;
    std::vector<std::vector<double>> trace_brlen_;
    std::vector<std::string> trace_newick_;
};

}  // namespace pyml::core

#endif  // PYML_CORE_MCMC_HPP
