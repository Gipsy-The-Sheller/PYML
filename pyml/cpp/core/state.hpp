#ifndef PYML_CORE_STATE_HPP
#define PYML_CORE_STATE_HPP

/* ===========================================================================
 *  core/state.hpp — the MCMC state and the likelihood it feeds
 *
 *  The state is exactly what a move may touch: the tree (topology links and
 *  branch lengths) and the stationary frequencies.  Everything else -- the
 *  alignment, the number of states, the tip names -- is constant for the run.
 *
 *  Snapshotting is the point of this file.  pyml's Python MCMC deep-copies the
 *  whole PhyloData every iteration, which drags along the pandas alignment,
 *  the model, and every node's (nstates x nsites) likelihood vector.  None of
 *  that can change under a move, so none of it is copied here: a Snapshot is
 *  the node linkage (4 ints each) plus one double per branch, a few KB for a
 *  hundred taxa.  Restoring is sufficient because LogLikPruner() recomputes
 *  every internal node on every call -- it holds no cache across evaluations
 *  -- and the leaf likelihoods depend only on the alignment, not on the tree.
 * =========================================================================== */

#include "tree.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyml::core {

/* ------------------------------------------------------------- data --- */

/// An alignment in the form the pruner wants: per tip, one state index per
/// site.  Characters that are not states map to -1, which the pruner turns
/// into a flat likelihood over all states.
struct DnaData {
    int n_taxa = 0;
    int n_sites = 0;
    int n_states = 4;
    std::vector<std::vector<int>> tip_state;   // [tip][site]
    std::vector<std::string> tip_name;

    /// `sequences[i]` belongs to `names[i]`.  Unknown characters become -1.
    [[nodiscard]] static DnaData FromStrings(const std::vector<std::string>& names,
                                             const std::vector<std::string>& sequences,
                                             const std::string& alphabet = "ACGT") {
        if (names.size() != sequences.size())
            throw std::invalid_argument("names and sequences differ in length");
        if (names.empty()) throw std::invalid_argument("no sequences");
        DnaData d;
        d.n_taxa = static_cast<int>(names.size());
        d.n_sites = static_cast<int>(sequences.front().size());
        d.n_states = static_cast<int>(alphabet.size());
        d.tip_name = names;
        d.tip_state.resize(names.size());
        for (std::size_t t = 0; t < sequences.size(); ++t) {
            if (static_cast<int>(sequences[t].size()) != d.n_sites)
                throw std::invalid_argument("sequences differ in length");
            auto& row = d.tip_state[t];
            row.resize(static_cast<std::size_t>(d.n_sites), -1);
            for (int s = 0; s < d.n_sites; ++s) {
                const auto pos = alphabet.find(sequences[t][static_cast<std::size_t>(s)]);
                row[static_cast<std::size_t>(s)] =
                    pos == std::string::npos ? -1 : static_cast<int>(pos);
            }
        }
        return d;
    }

    /// The pruner's Alignment type, built fresh.  Cheap enough to keep a copy.
    [[nodiscard]] PYML::LikelihoodCalculator<Real>::Alignment ToAlignment() const {
        return tip_state;
    }
};

/* -------------------------------------------------------- snapshot --- */

/// Everything a move can change, in a form that is cheap to copy and cheap to
/// put back.  Deliberately excludes per-node likelihood vectors.
struct Snapshot {
    std::vector<int> parent;
    std::vector<int> child;      // NODE_COUNT * Node::MAX_CHILDREN
    std::vector<int> n_child;
    std::vector<double> brlen;
    std::vector<double> freqs;
    bool valid = false;

    void Capture(const Tree& t, const std::vector<double>& pi) {
        const std::size_t n = static_cast<std::size_t>(t.NODE_COUNT);
        parent.resize(n);
        child.resize(n * PYML::Node<Real>::MAX_CHILDREN);
        n_child.resize(n);
        brlen.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto& nd = t.Nodes[i];
            parent[i] = nd.parent_idx;
            n_child[i] = nd.n_child;
            brlen[i] = nd.branch_length;
            for (int k = 0; k < PYML::Node<Real>::MAX_CHILDREN; ++k)
                child[i * PYML::Node<Real>::MAX_CHILDREN + static_cast<std::size_t>(k)] =
                    nd.child_idx[static_cast<std::size_t>(k)];
        }
        freqs = pi;
        valid = true;
    }

    /// Put the tree back, and return the frequencies to hand to Core::SetFreqs.
    /// Returning them rather than writing through a reference is deliberate:
    /// restoring frequencies has to refactor the rate matrix, and a caller
    /// that just assigns to a vector would silently skip that.
    [[nodiscard]] std::vector<double> Restore(Tree& t) const {
        if (!valid) throw std::logic_error("Snapshot::Restore on an empty snapshot");
        for (std::size_t i = 0; i < parent.size(); ++i) {
            auto& nd = t.Nodes[i];
            nd.parent_idx = parent[i];
            nd.n_child = n_child[i];
            nd.branch_length = brlen[i];
            for (int k = 0; k < PYML::Node<Real>::MAX_CHILDREN; ++k)
                nd.child_idx[static_cast<std::size_t>(k)] =
                    child[i * PYML::Node<Real>::MAX_CHILDREN + static_cast<std::size_t>(k)];
        }
        return freqs;
    }
};

/* ------------------------------------------------------------ core --- */

/// Smallest stationary frequency the core will build a rate matrix from.
///
/// This is not cosmetic.  Q's eigen decomposition does not survive a
/// degenerate frequency vector, and a Dirichlet(1,1,1,1) prior -- which is what
/// a default MrBayes DNA run puts on Pi -- is *uniform* on the simplex, so it
/// does not push the chain away from the boundary.  A proposal that drives a
/// frequency to zero would otherwise take the whole run down with an exception
/// from inside SetFreqs.
///
/// The floor is therefore part of the target, not just a guard: DirichletPrior
/// returns -inf below it, and every frequency move rejects below it.  Both
/// sides have to agree or the chain samples a different distribution than the
/// one it is checked against.  Reaching 1e-8 under a uniform prior has
/// probability around 1e-8, so the truncation is immaterial to the posterior.
inline constexpr double kMinFreq = 1e-6;

/// Build a rate matrix in the GTR sense with equal exchangeabilities (Nst=1 in
/// MrBayes terms), normalised so the expected substitution rate is one:
///     Q[i][j] = pi[j]  (i != j),  Q[i][i] = -(1 - pi[i])
///     then Q /= -sum_i pi[i] Q[i][i]
[[nodiscard]] inline std::vector<Real> BuildQ(const std::vector<double>& pi) {
    const int n = static_cast<int>(pi.size());
    double total = 0.0;
    for (double p : pi) total += p;
    if (n < 2 || total <= 0.0) throw std::invalid_argument("BuildQ: bad frequencies");
    for (double p : pi)
        if (p < kMinFreq) throw std::invalid_argument("BuildQ: frequency below kMinFreq");
    std::vector<Real> q(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) q[static_cast<std::size_t>(i) * n + j] = pi[static_cast<std::size_t>(j)];
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j)
            if (i != j) s += q[static_cast<std::size_t>(i) * n + j];
        q[static_cast<std::size_t>(i) * n + i] = -s;
    }
    double rate = 0.0;
    for (int i = 0; i < n; ++i)
        rate -= pi[static_cast<std::size_t>(i)] * q[static_cast<std::size_t>(i) * n + i];
    if (rate <= 0.0) throw std::invalid_argument("BuildQ: degenerate rate matrix");
    for (auto& v : q) v /= rate;
    return q;
}

/// Owns the tree, the model and the alignment, and hands out the likelihood.
/// Moves mutate the tree through here and the model is refactored whenever the
/// frequencies change.
class Core {
public:
    Core(const DnaData& data, Tree tree, std::vector<double> freqs)
        : data_(data),
          freqs_(CheckFreqs(data_, std::move(freqs))),
          calc_(CheckTree(data_, std::move(tree)), MakeModel(freqs_), data_.ToAlignment()) {}

    [[nodiscard]] Tree& tree() noexcept { return calc_.topology(); }
    [[nodiscard]] const Tree& tree() const noexcept { return calc_.topology(); }
    [[nodiscard]] const std::vector<double>& freqs() const noexcept { return freqs_; }
    [[nodiscard]] const DnaData& data() const noexcept { return data_; }
    [[nodiscard]] int n_states() const noexcept { return data_.n_states; }

    /// Replace the stationary frequencies and refactor the rate matrix.
    /// Throws if the frequencies are unusable; prefer TrySetFreqs in a move.
    void SetFreqs(std::vector<double> pi) {
        if (static_cast<int>(pi.size()) != data_.n_states)
            throw std::invalid_argument("frequencies do not match the number of states");
        freqs_ = std::move(pi);
        calc_.model().InitWithQPi(BuildQ(freqs_), freqs_);
    }

    /// Transactional form, for proposals: on failure the state is left exactly
    /// as it was and false comes back, so the caller can reject the proposal
    /// rather than bring the run down.
    ///
    /// An extreme frequency vector can make Q's eigen decomposition fail to
    /// converge -- eigGeneral() reports NotConverged and InitWithQPi throws.
    /// Under a Dirichlet(1,1,1,1) prior, which does not repel the chain from
    /// the simplex boundary, the chain does occasionally propose such a point.
    /// It is a bad proposal, not a bug, so it is rejected like any other.
    [[nodiscard]] bool TrySetFreqs(std::vector<double> pi) {
        if (static_cast<int>(pi.size()) != data_.n_states) return false;
        for (double p : pi)
            if (!(p >= kMinFreq)) return false;
        try {
            calc_.model().InitWithQPi(BuildQ(pi), pi);
        } catch (const std::exception&) {
            // rebuild from the frequencies we are keeping, so the model is
            // left consistent rather than half-updated
            calc_.model().InitWithQPi(BuildQ(freqs_), freqs_);
            return false;
        }
        freqs_ = std::move(pi);
        return true;
    }

    /// Rebuild the model from the current frequencies.
    void RefreshModel() { calc_.model().InitWithQPi(BuildQ(freqs_), freqs_); }

    /// Log likelihood of the current tree and frequencies.  Recomputes every
    /// internal node; there is no cache to invalidate.
    [[nodiscard]] double LogLik() { return calc_.LogLikPruner(); }

    /// Total tree length, the statistic a GammaDir prior acts on.
    [[nodiscard]] double TreeLength() const {
        double total = 0.0;
        const int root = Root(tree());
        for (int v = 0; v < tree().NODE_COUNT; ++v)
            if (v != root) total += At(tree(), v).branch_length;
        return total;
    }

private:
    // Validated in the initialiser list, so the checks run before anything is
    // built from them.  Members are initialised in declaration order, which is
    // why data_ must be declared first.
    [[nodiscard]] static std::vector<double> CheckFreqs(const DnaData& d,
                                                        std::vector<double> pi) {
        if (static_cast<int>(pi.size()) != d.n_states)
            throw std::invalid_argument("frequencies do not match the number of states");
        return pi;
    }

    [[nodiscard]] static Tree CheckTree(const DnaData& d, Tree t) {
        if (t.TAXA_COUNT != d.n_taxa)
            throw std::invalid_argument("tree and alignment have different taxon counts");
        return t;
    }

    [[nodiscard]] static PYML::SubstMatrix<Real> MakeModel(const std::vector<double>& pi) {
        PYML::SubstMatrix<Real> m;
        m.InitWithQPi(BuildQ(pi), pi);
        return m;
    }

    DnaData data_;
    std::vector<double> freqs_;
    PYML::LikelihoodCalculator<Real> calc_;
};

}  // namespace pyml::core

#endif  // PYML_CORE_STATE_HPP
