#ifndef PYML_CPP_LIKELIHOOD_HPP
#define PYML_CPP_LIKELIHOOD_HPP

#include "evd.hpp"
#include "vectorize.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace PYML {

template <VecMnpUtils::Vectorable Precision>
struct Node {
    int parent_idx{-1};
    int cleft_idx{-1};
    int cright_idx{-1};
    Precision branch_length{};
    std::vector<Precision> likelihood;

    [[nodiscard]] bool IsTip() const noexcept {
        return cleft_idx < 0 && cright_idx < 0;
    }
    [[nodiscard]] bool IsRoot() const noexcept { return parent_idx < 0; }
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

    void connect(int parent, int left, int right) {
        if (parent < 0 || left < 0 || right < 0 ||
            parent >= NODE_COUNT || left >= NODE_COUNT || right >= NODE_COUNT)
            throw std::out_of_range("tree node index out of range");
        Nodes[parent].cleft_idx = left;
        Nodes[parent].cright_idx = right;
        Nodes[left].parent_idx = parent;
        Nodes[right].parent_idx = parent;
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

    [[nodiscard]] std::vector<Precision> PMatrix(Precision t) const {
        if (!initialized) throw std::logic_error("substitution matrix is not initialized");
        const auto p = mpe::linalg::transitionMatrix(eigen, t);
        return {p.data().begin(), p.data().end()};
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
        initialise_leaves();
    }

    [[nodiscard]] Precision LogLikPruner() {
        if (!model_.initialized) throw std::logic_error("substitution model is not initialized");
        if (topology_.NODE_COUNT == 0) throw std::logic_error("tree is empty");
        const int root = topology_.root_index();
        compute_node(root);
        Precision result{};
        const auto& root_likelihood = topology_.Nodes[static_cast<std::size_t>(root)].likelihood;
        for (std::size_t site = 0; site < sites_; ++site) {
            Precision site_likelihood{};
            for (int state = 0; state < model_.states; ++state)
                site_likelihood += model_.Pis[static_cast<std::size_t>(state)] *
                    root_likelihood[static_cast<std::size_t>(state) * sites_ + site];
            result += std::log(std::max(site_likelihood, Precision{1e-300}));
        }
        return result;
    }

    BinaryTree<Precision>& topology() noexcept { return topology_; }
    const SubstMatrix<Precision>& model() const noexcept { return model_; }

private:
    BinaryTree<Precision> topology_;
    SubstMatrix<Precision> model_;
    Alignment taxastates_;
    std::size_t sites_{};

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

    void compute_node(int index) {
        auto& node = topology_.Nodes[static_cast<std::size_t>(index)];
        if (node.IsTip()) return;
        compute_node(node.cleft_idx);
        compute_node(node.cright_idx);
        const auto left_p = model_.PMatrix(topology_.Nodes[static_cast<std::size_t>(node.cleft_idx)].branch_length);
        const auto right_p = model_.PMatrix(topology_.Nodes[static_cast<std::size_t>(node.cright_idx)].branch_length);
        const auto& left = topology_.Nodes[static_cast<std::size_t>(node.cleft_idx)].likelihood;
        const auto& right = topology_.Nodes[static_cast<std::size_t>(node.cright_idx)].likelihood;
        node.likelihood.assign(sites_ * static_cast<std::size_t>(model_.states), Precision{});
        std::vector<Precision> left_sum(sites_);
        std::vector<Precision> right_sum(sites_);
        for (int parent_state = 0; parent_state < model_.states; ++parent_state) {
            std::fill(left_sum.begin(), left_sum.end(), Precision{});
            std::fill(right_sum.begin(), right_sum.end(), Precision{});
            for (int child_state = 0; child_state < model_.states; ++child_state) {
                VecMnpUtils::MultiplyAccumulate(
                    left_sum.data(),
                    left.data() + static_cast<std::size_t>(child_state) * sites_,
                    left_p[static_cast<std::size_t>(parent_state * model_.states + child_state)],
                    static_cast<int>(sites_));
                VecMnpUtils::MultiplyAccumulate(
                    right_sum.data(),
                    right.data() + static_cast<std::size_t>(child_state) * sites_,
                    right_p[static_cast<std::size_t>(parent_state * model_.states + child_state)],
                    static_cast<int>(sites_));
            }
            for (std::size_t site = 0; site < sites_; ++site) {
                node.likelihood[static_cast<std::size_t>(parent_state) * sites_ + site] =
                    left_sum[site] * right_sum[site];
            }
        }
    }
};

}  // namespace PYML

#endif
