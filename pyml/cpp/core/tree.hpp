#ifndef PYML_CORE_TREE_HPP
#define PYML_CORE_TREE_HPP

/* ===========================================================================
 *  core/tree.hpp — unrooted binary trees for the MCMC core
 *
 *  An unrooted N-taxon binary tree has 2N-2 nodes (N tips, N-2 internal) and
 *  2N-3 edges.  We store it as a rooted tree whose root is an internal node of
 *  degree three — the representation MrBayes uses, and the one Felsenstein's
 *  pruning is indifferent to (the likelihood does not depend on where the tree
 *  is rooted).
 *
 *  This is exactly PYML::BinaryTree's layout, so the pruning code takes it
 *  with no conversion.  Node numbering:
 *
 *      0 .. N-1      tips
 *      N .. 2N-3     internal
 *      root          degree three, found with Root()
 *
 *  Branch lengths are indexed by node: every non-root node carries exactly one
 *  branch, the one above it.  tree.Nodes[root].branch_length is unused.
 * =========================================================================== */

#include "../likelihood.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyml::core {

using Real = double;
using Tree = PYML::BinaryTree<Real>;
using Node = PYML::Node<Real>;

[[nodiscard]] inline bool IsTip(const Tree& t, int node) noexcept {
    return node < t.TAXA_COUNT;
}

[[nodiscard]] inline int Root(const Tree& t) { return t.root_index(); }

/// Number of edges, i.e. of usable branch lengths.
[[nodiscard]] inline int BranchCount(const Tree& t) noexcept {
    return t.TAXA_COUNT > 0 ? 2 * t.TAXA_COUNT - 3 : 0;
}

[[nodiscard]] inline Node& At(Tree& t, int node) {
    return t.Nodes[static_cast<std::size_t>(node)];
}
[[nodiscard]] inline const Node& At(const Tree& t, int node) {
    return t.Nodes[static_cast<std::size_t>(node)];
}

/* -------------------------------------------------- link primitives --- */

/// Remove `child` from `parent`'s child list, closing the gap.
inline void DetachChild(Tree& t, int parent, int child) {
    auto& p = At(t, parent);
    for (int s = 0; s < p.n_child; ++s)
        if (p.child_idx[static_cast<std::size_t>(s)] == child) {
            for (int k = s; k + 1 < p.n_child; ++k)
                p.child_idx[static_cast<std::size_t>(k)] =
                    p.child_idx[static_cast<std::size_t>(k + 1)];
            p.child_idx[static_cast<std::size_t>(--p.n_child)] = -1;
            At(t, child).parent_idx = -1;
            return;
        }
    throw std::logic_error("DetachChild: not a child of that parent");
}

/// Add `child` to `parent`'s child list and set the child's parent link.
/// `child` must not currently be attached anywhere.
inline void AppendChild(Tree& t, int parent, int child) {
    auto& p = At(t, parent);
    if (p.n_child >= static_cast<int>(Node::MAX_CHILDREN))
        throw std::logic_error("AppendChild: parent already has the maximum children");
    p.child_idx[static_cast<std::size_t>(p.n_child++)] = child;
    At(t, child).parent_idx = parent;
}

/// Move the subtree rooted at `node` under `new_parent`.
inline void Reparent(Tree& t, int node, int new_parent) {
    const int old_parent = At(t, node).parent_idx;
    if (old_parent < 0) throw std::logic_error("Reparent: cannot move the root");
    DetachChild(t, old_parent, node);
    AppendChild(t, new_parent, node);
}

/* --------------------------------------------------------- checks --- */

/// Structural check.  Throws on anything that would make the pruner misbehave.
/// For tests and for the boundary where a tree enters the core, not for the
/// MCMC inner loop.
inline void Validate(const Tree& t) {
    if (t.TAXA_COUNT < 3) throw std::invalid_argument("tree needs at least 3 taxa");
    if (t.NODE_COUNT != 2 * t.TAXA_COUNT - 2)
        throw std::invalid_argument("unrooted node count must be 2N-2");

    int roots = 0;
    for (int i = 0; i < t.NODE_COUNT; ++i) {
        const auto& n = At(t, i);
        if (n.n_child < 0 || n.n_child > static_cast<int>(Node::MAX_CHILDREN))
            throw std::invalid_argument("bad child count");
        if (n.IsRoot()) ++roots;
        if (IsTip(t, i) != (n.n_child == 0))
            throw std::invalid_argument("leaf/internal node has the wrong degree");
        for (int s = 0; s < n.n_child; ++s) {
            const int c = n.child_idx[static_cast<std::size_t>(s)];
            if (c < 0 || c >= t.NODE_COUNT) throw std::invalid_argument("child out of range");
            if (At(t, c).parent_idx != i)
                throw std::invalid_argument("parent/child link is inconsistent");
        }
    }
    if (roots != 1) throw std::invalid_argument("tree must have exactly one root");

    // Degree three everywhere in the unrooted sense: the root has three
    // children and no parent, every other internal node has two and one.
    const int r = Root(t);
    if (At(t, r).n_child != 3)
        throw std::invalid_argument("the root of an unrooted tree must have three children");
    for (int i = t.TAXA_COUNT; i < t.NODE_COUNT; ++i)
        if (i != r && At(t, i).n_child != 2)
            throw std::invalid_argument("an internal node must have two children");
}

/// A topology-independent label for the branch above `v`, and for the branch
/// above any node, keyed on tips rather than on node numbers.
///
/// NNI only relinks nodes; it never renumbers them.  So "the branch above node
/// 7" is a different physical branch before and after a topology move, and a
/// trace of brlen[7] would be a trace of a moving target.  A branch in an
/// unrooted tree is identified by the split it induces, and a split can be
/// named by the smallest tip index on the side that does not contain tip 0.
/// Two distinct branches always get distinct labels, and the label of a given
/// branch does not change when the topology moves.
[[nodiscard]] inline int BranchLabel(const Tree& t, int v) {
    if (v == Root(t)) throw std::invalid_argument("the root has no branch of its own");

    // smallest tip at or below each node, walking up from the tips
    std::vector<int> lowest(static_cast<std::size_t>(t.NODE_COUNT), -1);
    for (int i = 0; i < t.TAXA_COUNT; ++i) lowest[static_cast<std::size_t>(i)] = i;
    for (int i = t.TAXA_COUNT; i < t.NODE_COUNT; ++i) {
        const auto& n = At(t, i);
        int lo = -1;
        for (int s = 0; s < n.n_child; ++s) {
            const int c = lowest[static_cast<std::size_t>(n.child_idx[static_cast<std::size_t>(s)])];
            if (c >= 0 && (lo < 0 || c < lo)) lo = c;
        }
        lowest[static_cast<std::size_t>(i)] = lo;
    }

    const int below = lowest[static_cast<std::size_t>(v)];
    if (below != 0) return below;              // tip 0 is on the other side

    // tip 0 is below v, so name the branch by the smallest tip that is not.
    std::vector<char> under(static_cast<std::size_t>(t.NODE_COUNT), 0);
    std::vector<int> stack{v};
    while (!stack.empty()) {
        const int n = stack.back();
        stack.pop_back();
        under[static_cast<std::size_t>(n)] = 1;
        const auto& nd = At(t, n);
        for (int s = 0; s < nd.n_child; ++s)
            stack.push_back(nd.child_idx[static_cast<std::size_t>(s)]);
    }
    for (int i = 1; i < t.TAXA_COUNT; ++i)
        if (!under[static_cast<std::size_t>(i)]) return i;
    throw std::logic_error("BranchLabel: every tip is on one side of the branch");
}

/// Branch lengths in a fixed order: the branches sorted by BranchLabel, so
/// column k means the same physical branch at every sample.  Returns 2N-3
/// values.
[[nodiscard]] inline std::vector<double> BranchLengthsCanonical(const Tree& t) {
    const int root = Root(t);
    std::vector<std::pair<int, double>> labelled;
    labelled.reserve(static_cast<std::size_t>(BranchCount(t)));
    for (int v = 0; v < t.NODE_COUNT; ++v) {
        if (v == root) continue;
        labelled.emplace_back(BranchLabel(t, v), At(t, v).branch_length);
    }
    std::sort(labelled.begin(), labelled.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<double> out;
    out.reserve(labelled.size());
    for (const auto& kv : labelled) out.push_back(kv.second);
    return out;
}

/// Edges with an internal node at both ends.  An unrooted binary tree over
/// N >= 3 tips has N-3 of them, and they are exactly the edges an NNI can act
/// on: every other edge has a tip at one end and admits no rearrangement.
[[nodiscard]] inline std::vector<int> InternalEdges(const Tree& t) {
    std::vector<int> edges;
    for (int v = 0; v < t.NODE_COUNT; ++v) {
        if (IsTip(t, v)) continue;
        const int u = At(t, v).parent_idx;
        if (u >= 0 && !IsTip(t, u)) edges.push_back(v);
    }
    return edges;
}

/* ------------------------------------------------------ construction --- */

/// A caterpillar: the root carries tips N-2 and N-1 plus a chain that ends in
/// the cherry (0,1).  A valid starting point for any N >= 3.
[[nodiscard]] inline Tree MakeStar(int n_taxa, Real brlen) {
    if (n_taxa < 3) throw std::invalid_argument("MakeStar needs at least 3 taxa");
    Tree t = Tree::Unrooted(n_taxa);

    if (n_taxa == 3) {
        t.connect(n_taxa, {0, 1, 2});
    } else {
        const int chain0 = n_taxa;                  // the cherry (0,1)
        const int root = 2 * n_taxa - 3;            // the last internal node
        t.connect(chain0, {0, 1});
        int prev = chain0;
        for (int k = n_taxa + 1; k <= 2 * n_taxa - 4; ++k) {
            t.connect(k, {prev, k - n_taxa + 1});   // adds tip 2, 3, ..., N-3
            prev = k;
        }
        t.connect(root, {n_taxa - 2, n_taxa - 1, prev});
    }
    const int r = Root(t);
    for (int v = 0; v < t.NODE_COUNT; ++v)
        if (v != r) At(t, v).branch_length = brlen;
    Validate(t);
    return t;
}

/* ------------------------------------------------------------ Newick --- */

namespace detail {

struct NewickNode {
    std::string name;
    double brlen = 0.0;
    std::vector<std::unique_ptr<NewickNode>> kids;
};

struct NewickParser {
    const std::string& s;
    std::size_t i = 0;
    explicit NewickParser(const std::string& str) : s(str) {}

    void SkipSpace() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }
    void Expect(char c) {
        SkipSpace();
        if (i >= s.size() || s[i] != c)
            throw std::invalid_argument(std::string("Newick: expected '") + c + "'");
        ++i;
    }
    [[nodiscard]] bool Peek(char c) {
        SkipSpace();
        return i < s.size() && s[i] == c;
    }

    [[nodiscard]] std::string ReadLabel() {
        SkipSpace();
        if (i < s.size() && s[i] == '\'') {          // quoted label
            ++i;
            std::string out;
            while (i < s.size() && s[i] != '\'') out.push_back(s[i++]);
            Expect('\'');
            return out;
        }
        std::string out;
        while (i < s.size() && !std::isspace(static_cast<unsigned char>(s[i])) &&
               s[i] != ',' && s[i] != ')' && s[i] != '(' && s[i] != ':' && s[i] != ';')
            out.push_back(s[i++]);
        return out;
    }

    [[nodiscard]] double ReadLength() {
        Expect(':');
        SkipSpace();
        const char* start = s.c_str() + i;
        char* end = nullptr;
        const double v = std::strtod(start, &end);
        if (end == start) throw std::invalid_argument("Newick: malformed branch length");
        i += static_cast<std::size_t>(end - start);
        return v;
    }

    std::unique_ptr<NewickNode> ParseSubtree() {
        SkipSpace();
        auto node = std::make_unique<NewickNode>();
        if (Peek('(')) {
            Expect('(');
            for (;;) {
                node->kids.push_back(ParseSubtree());
                if (Peek(',')) { Expect(','); continue; }
                break;
            }
            Expect(')');
            node->name = ReadLabel();               // internal labels are dropped
        } else {
            node->name = ReadLabel();
        }
        if (Peek(':')) node->brlen = ReadLength();
        return node;
    }

    [[nodiscard]] std::unique_ptr<NewickNode> Parse() {
        auto r = ParseSubtree();
        SkipSpace();
        if (i < s.size() && s[i] == ';') ++i;
        return r;
    }
};

/// Bottom-up flattening.  Tips take 0..N-1 in the order met, internals follow,
/// so every parent's index exceeds its children's.  The last internal node
/// assigned is the root, which is why `root_index` is 2N-3.
inline int Flatten(const NewickNode* n, Tree& t, std::vector<std::string>& names,
                   int& next_tip, int& next_internal, int root_index) {
    std::vector<int> kids;
    kids.reserve(n->kids.size());
    for (const auto& k : n->kids)
        kids.push_back(Flatten(k.get(), t, names, next_tip, next_internal, root_index));

    int idx;
    if (kids.empty()) {
        idx = next_tip++;
        names[static_cast<std::size_t>(idx)] = n->name;
    } else {
        idx = next_internal++;
        t.connect(idx, kids);
    }
    if (idx != root_index) At(t, idx).branch_length = n->brlen;
    return idx;
}

}  // namespace detail

/// Parse a Newick string into an unrooted tree.  Tip names, if requested, come
/// back in node order, so `names[i]` is the label of tip i.
///
/// A basal trifurcation `(a,b,(c,d));` is taken as already unrooted.  A rooted
/// `((a,b),(c,d));` is unrooted on the branch below the first child of the
/// root: that child becomes the new root, and the two branch lengths involved
/// merge into the single edge that replaces them.
[[nodiscard]] inline Tree FromNewick(const std::string& newick,
                                     std::vector<std::string>* tip_names = nullptr) {
    detail::NewickParser parser(newick);
    auto root = parser.Parse();
    if (!root) throw std::invalid_argument("Newick: empty");

    if (root->kids.size() == 2) {
        auto& x = root->kids[0];
        auto& y = root->kids[1];
        if (x->kids.empty() && y->kids.empty())
            throw std::invalid_argument("Newick: root has two tips");
        // Reroot at whichever child is internal.  If both are, either works;
        // taking x leaves x with three children, which is what an unrooted
        // root must have.
        auto& keep = x->kids.empty() ? y : x;
        auto& drop = x->kids.empty() ? x : y;
        if (keep->kids.size() >= 3)
            throw std::invalid_argument("Newick: cannot unroot, too many children at the root");
        drop->brlen += keep->brlen;             // merge the two edges into one
        keep->kids.push_back(std::move(drop));
        root = std::move(keep);
    }
    if (root->kids.size() != 3)
        throw std::invalid_argument("Newick: an unrooted tree needs a trifurcating root");

    int n_tips = 0;
    {
        std::vector<const detail::NewickNode*> stack{root.get()};
        while (!stack.empty()) {
            const auto* n = stack.back();
            stack.pop_back();
            if (n->kids.empty()) {
                if (n->name.empty()) throw std::invalid_argument("Newick: unnamed tip");
                ++n_tips;
            }
            for (const auto& k : n->kids) stack.push_back(k.get());
        }
    }
    if (n_tips < 3) throw std::invalid_argument("Newick: need at least 3 taxa");

    Tree t = Tree::Unrooted(n_tips);
    std::vector<std::string> names(static_cast<std::size_t>(n_tips));
    int next_tip = 0, next_internal = n_tips;
    const int root_index = 2 * n_tips - 3;
    const int got = detail::Flatten(root.get(), t, names, next_tip, next_internal, root_index);
    if (got != root_index || next_tip != n_tips)
        throw std::invalid_argument("Newick: inconsistent tree");

    Validate(t);
    if (tip_names) *tip_names = std::move(names);
    return t;
}

/// Newick text for an unrooted tree.  The root comes out as a basal
/// trifurcation, which is the standard unrooted output form.
[[nodiscard]] inline std::string ToNewick(const Tree& t,
                                          const std::vector<std::string>& tip_names) {
    const int root = Root(t);
    std::string out;

    std::function<void(int)> write = [&](int node) {
        const auto& n = At(t, node);
        if (n.n_child > 0) {
            out += '(';
            for (int s = 0; s < n.n_child; ++s) {
                if (s) out += ',';
                write(n.child_idx[static_cast<std::size_t>(s)]);
            }
            out += ')';
        } else if (IsTip(t, node)) {
            const std::string& nm = tip_names[static_cast<std::size_t>(node)];
            const bool plain = !nm.empty() &&
                nm.find_first_of(" \t(),:;'") == std::string::npos;
            out += plain ? nm : ("'" + nm + "'");
        }
        if (node != root) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), ":%.10g", At(t, node).branch_length);
            out += buf;
        }
    };
    write(root);
    out += ';';
    return out;
}

}  // namespace pyml::core

#endif  // PYML_CORE_TREE_HPP
