#ifndef PYML_CPP_LIKELIHOOD_HPP
#define PYML_CPP_LIKELIHOOD_HPP

#include "evd.hpp"
#include "vectorize.hpp"

#include <cmath>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace PYML {

/// One node of a rooted tree.  Up to three children: a binary root has two,
/// and rooting an *unrooted* tree at an internal node makes that node
/// trifurcating, which is how MrBayes-style unrooted trees are represented.
/// Felsenstein's pruning does not care how many children a node has -- PAML's
/// ConditionalPNode() loops over nodes[inode].nson the same way.
template <VecMnpUtils::Vectorable Precision>
struct Node {
    static constexpr int MAX_CHILDREN = 3;
    int parent_idx{-1};
    std::array<int, MAX_CHILDREN> child_idx{-1, -1, -1};
    int n_child{0};
    Precision branch_length{};
    std::vector<Precision> likelihood;

    [[nodiscard]] bool IsTip() const noexcept { return n_child == 0; }
    [[nodiscard]] bool IsRoot() const noexcept { return parent_idx < 0; }

    // Kept so existing two-child code reads unchanged.
    [[nodiscard]] int cleft_idx() const noexcept { return child_idx[0]; }
    [[nodiscard]] int cright_idx() const noexcept { return child_idx[1]; }
};

template <VecMnpUtils::Vectorable Precision>
struct BinaryTree {
    int TAXA_COUNT{};
    int NODE_COUNT{};
    int BRANCH_COUNT{};
    std::vector<Node<Precision>> Nodes;

    explicit BinaryTree(int taxa_count = 0)
        : TAXA_COUNT(taxa_count),
          NODE_COUNT(taxa_count > 0 ? 2 * taxa_count - 1 : 0),
          BRANCH_COUNT(taxa_count > 0 ? 2 * taxa_count - 2 : 0),
          Nodes(static_cast<std::size_t>(NODE_COUNT)) {
        if (taxa_count < 0) throw std::invalid_argument("taxa_count must be non-negative");
    }

    [[nodiscard]] int root_index() const {
        for (int i = 0; i < NODE_COUNT; ++i)
            if (Nodes[static_cast<std::size_t>(i)].IsRoot()) return i;
        throw std::invalid_argument("tree has no root");
    }

    void connect(int parent, const int* kids, int count) {
        if (parent < 0 || parent >= NODE_COUNT)
            throw std::out_of_range("tree node index out of range");
        if (count < 0 || count > static_cast<int>(Node<Precision>::MAX_CHILDREN))
            throw std::invalid_argument("too many children for one node");
        auto& p = Nodes[static_cast<std::size_t>(parent)];
        p.n_child = 0;
        for (int s = 0; s < count; ++s) {
            const int k = kids[s];
            if (k < 0 || k >= NODE_COUNT) throw std::out_of_range("tree node index out of range");
            p.child_idx[static_cast<std::size_t>(p.n_child++)] = k;
            Nodes[static_cast<std::size_t>(k)].parent_idx = parent;
        }
    }

    void connect(int parent, std::initializer_list<int> kids) {
        connect(parent, kids.begin(), static_cast<int>(kids.size()));
    }

    void connect(int parent, const std::vector<int>& kids) {
        connect(parent, kids.data(), static_cast<int>(kids.size()));
    }

    void connect(int parent, int left, int right) { connect(parent, {left, right}); }

    /// An *unrooted* N-taxon tree, stored as a rooted tree whose root is an
    /// internal node of degree three.  Node count is 2N-2 (N tips, N-2
    /// internal) and edge count 2N-3 -- one fewer than the rooted layout
    /// above, because the root has no branch of its own.  Every non-root node
    /// carries exactly one branch, the one above it, so branch lengths are
    /// indexed by node and brlen[root] is unused.
    ///
    /// This is how MrBayes represents unrooted trees, and Felsenstein's
    /// pruning is indifferent to where the tree is rooted, so the existing
    /// likelihood code takes this layout unchanged.
    [[nodiscard]] static BinaryTree Unrooted(int taxa_count) {
        if (taxa_count < 1) throw std::invalid_argument("taxa_count must be positive");
        BinaryTree t;
        t.TAXA_COUNT = taxa_count;
        t.NODE_COUNT = 2 * taxa_count - 2;
        t.BRANCH_COUNT = 2 * taxa_count - 3;
        t.Nodes.resize(static_cast<std::size_t>(t.NODE_COUNT));
        return t;
    }
};

template <VecMnpUtils::Vectorable Precision>
struct SubstMatrix {
    int states{};
    std::vector<Precision> QMatrix;
    std::vector<Precision> Pis;
    mpe::linalg::EigenSystem<Precision> eigen;
    bool initialized{false};

    void InitWithQPi(std::vector<Precision> qmatrix, std::vector<Precision> pis) {
        if (pis.empty() || qmatrix.size() != pis.size() * pis.size())
            throw std::invalid_argument("Q matrix and stationary frequencies have incompatible sizes");
        states = static_cast<int>(pis.size());
        for (Precision pi : pis)
            if (!(pi >= Precision{})) throw std::invalid_argument("stationary frequencies must be non-negative");
        QMatrix = std::move(qmatrix);
        Pis = std::move(pis);
        mpe::linalg::Matrix<Precision> q(static_cast<std::size_t>(states),
                                         static_cast<std::size_t>(states));
        for (int i = 0; i < states; ++i)
            for (int j = 0; j < states; ++j)
                q(i, j) = QMatrix[static_cast<std::size_t>(i * states + j)];
        const auto diagnostics = mpe::linalg::eigGeneral(q, eigen);
        if (diagnostics.status != mpe::linalg::EigenStatus::Ok)
            throw std::runtime_error("eigenvalue decomposition of Q failed");
        initialized = true;
    }

    /// Writes P(t) into `out` (states x states, row-major), using `scratch`
    /// (same shape) as workspace.  Both belong to the caller.
    ///
    /// This deliberately does not return a fresh matrix.  It is called twice
    /// per internal node on every likelihood evaluation, and the previous
    /// version allocated two matrices inside transitionMatrix() plus a vector
    /// to return them in -- four heap allocations and a copy per branch, all
    /// of it pure overhead in the pruning inner loop.
    void PMatrix(Precision t, mpe::linalg::Matrix<Precision>& out,
                 mpe::linalg::Matrix<Precision>& scratch) const {
        if (!initialized) throw std::logic_error("substitution matrix is not initialized");
        mpe::linalg::transitionMatrix(eigen, t, Precision{1}, out, scratch);
    }
};

template <VecMnpUtils::Vectorable Precision>
class LikelihoodCalculator {
public:
    using Alignment = std::vector<std::vector<int>>;

    LikelihoodCalculator(BinaryTree<Precision> topology,
                         SubstMatrix<Precision> model,
                         Alignment taxastates)
        : topology_(std::move(topology)),
          model_(std::move(model)),
          taxastates_(std::move(taxastates)) {
        if (static_cast<int>(taxastates_.size()) != topology_.TAXA_COUNT)
            throw std::invalid_argument("alignment taxa count does not match tree");
        sites_ = taxastates_.empty() ? 0 : taxastates_.front().size();
        for (const auto& row : taxastates_)
            if (row.size() != sites_) throw std::invalid_argument("alignment rows have different lengths");

        // Workspace, sized once.  compute_node() runs once per internal node
        // per likelihood evaluation, so none of this may be allocated there.
        const std::size_t n = static_cast<std::size_t>(model_.states);
        pmat_ = mpe::linalg::Matrix<Precision>{n, n, Precision{}};
        pmat_scratch_ = mpe::linalg::Matrix<Precision>{n, n, Precision{}};
        root_lik_.assign(sites_, Precision{});
        initialise_leaves();
    }

    [[nodiscard]] Precision LogLikPruner() {
        if (!model_.initialized) throw std::logic_error("substitution model is not initialized");
        if (topology_.NODE_COUNT == 0) throw std::logic_error("tree is empty");
        const int root = topology_.root_index();
        compute_node(root);

        // Root reduction: sum over states of pi_j * L[j][site].  One pass along
        // the site axis per state -- the shape the vector kernels are built
        // for.  The previous form was a length-states dot product per site,
        // with the reduction in a scalar accumulator, and could not vectorize.
        const auto& root_likelihood = topology_.Nodes[static_cast<std::size_t>(root)].likelihood;
        std::fill(root_lik_.begin(), root_lik_.end(), Precision{});
        for (int state = 0; state < model_.states; ++state)
            VecMnpUtils::MultiplyAccumulate(
                root_lik_.data(),
                root_likelihood.data() + static_cast<std::size_t>(state) * sites_,
                model_.Pis[static_cast<std::size_t>(state)],
                static_cast<int>(sites_));

        Precision result{};
        for (std::size_t site = 0; site < sites_; ++site)
            result += std::log(std::max(root_lik_[site], Precision{1e-300}));
        return result;
    }

    BinaryTree<Precision>& topology() noexcept { return topology_; }
    const BinaryTree<Precision>& topology() const noexcept { return topology_; }
    const SubstMatrix<Precision>& model() const noexcept { return model_; }

    /// Non-const access, for callers that mutate the model between
    /// evaluations -- an MCMC that proposes new stationary frequencies, say.
    /// The eigen decomposition is refactored by InitWithQPi(), so a caller
    /// changing Q or the frequencies must call that rather than assigning to
    /// the members.
    SubstMatrix<Precision>& model() noexcept { return model_; }

private:
    BinaryTree<Precision> topology_;
    SubstMatrix<Precision> model_;
    Alignment taxastates_;
    std::size_t sites_{};

    // Workspace owned by the calculator rather than allocated per node.
    mpe::linalg::Matrix<Precision> pmat_;          // states x states
    mpe::linalg::Matrix<Precision> pmat_scratch_;  // same, for transitionMatrix
    std::vector<Precision> root_lik_;              // sites

    void initialise_leaves() {
        for (int taxon = 0; taxon < topology_.TAXA_COUNT; ++taxon) {
            auto& leaf = topology_.Nodes[static_cast<std::size_t>(taxon)];
            leaf.likelihood.assign(sites_ * static_cast<std::size_t>(model_.states), Precision{});
            for (std::size_t site = 0; site < sites_; ++site) {
                const int state = taxastates_[static_cast<std::size_t>(taxon)][site];
                auto* values = leaf.likelihood.data();
                if (state < 0 || state >= model_.states) {
                    for (int i = 0; i < model_.states; ++i)
                        values[static_cast<std::size_t>(i) * sites_ + site] =
                            Precision{1} / static_cast<Precision>(model_.states);
                } else {
                        values[static_cast<std::size_t>(state) * sites_ + site] = Precision{1};
                }
            }
        }
    }

    /// One Felsenstein partial:  parent[j][h] = sum_k P[j][k] * child[k][h],
    /// over a states-major pair of blocks.
    ///
    /// This is the kernel that matters.  Expressing the same update as
    /// states*states AXPYs -- one per (parent,child) state pair, which is what
    /// this used to do -- re-streams the whole child block and rewrites the
    /// whole accumulator for every pair.  The memory traffic is then
    /// `states` times higher for identical arithmetic, and at the 4 states
    /// phylogenetics uses that costs more than the 4-wide SIMD wins back.  It
    /// measured slower than the plain scalar loop.  The kernel below walks the
    /// site axis once, keeps the k-reduction in registers, and stores the
    /// parent block once.
    ///
    /// `overwrite` false multiplies into the existing parent, which is what a
    /// node's second child needs -- fused into the same pass rather than
    /// costing another sweep over the parent.
    void multiply_partial(Precision* parent, const Precision* child,
                          const Precision* pmat, bool overwrite) const {
        const auto sites = static_cast<long long>(sites_);
        const int states = model_.states;
        if constexpr (std::is_same_v<Precision, float>) {
            if (overwrite) VecMnpUtils::PmatMulF32(parent, child, pmat, states, sites);
            else           VecMnpUtils::PmatMulAccF32(parent, child, pmat, states, sites);
        } else {
            if (overwrite) VecMnpUtils::PmatMulF64(parent, child, pmat, states, sites);
            else           VecMnpUtils::PmatMulAccF64(parent, child, pmat, states, sites);
        }
    }

    void compute_node(int index) {
        auto& node = topology_.Nodes[static_cast<std::size_t>(index)];
        if (node.IsTip()) return;
        for (int slot = 0; slot < node.n_child; ++slot)
            compute_node(node.child_idx[static_cast<std::size_t>(slot)]);

        // resize, not assign: the kernel below writes every element of the
        // first child's contribution, so pre-filling is wasted work.  After
        // the first call the size is already right and this is a no-op.
        const std::size_t block = sites_ * static_cast<std::size_t>(model_.states);
        if (node.likelihood.size() != block) node.likelihood.resize(block);
        auto* parent = node.likelihood.data();

        // The first child seeds the parent and the rest multiply in.  A
        // trifurcating root -- an unrooted tree rooted at an internal node --
        // falls out of the same loop.
        for (int slot = 0; slot < node.n_child; ++slot) {
            const auto& child = topology_.Nodes[
                static_cast<std::size_t>(node.child_idx[static_cast<std::size_t>(slot)])];
            model_.PMatrix(child.branch_length, pmat_, pmat_scratch_);
            multiply_partial(parent, child.likelihood.data(),
                             pmat_.data().data(), slot == 0);
        }
    }
};

}  // namespace PYML

#endif
