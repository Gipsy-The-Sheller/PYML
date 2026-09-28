/* Benchmark + correctness anchor for PYML::LikelihoodCalculator.
 *
 * Two jobs:
 *   1. Check the pruner against an independent value.  The first case is the
 *      4-taxon JC69 tree whose exact likelihood can be written out by hand;
 *      -25.786608878551 was obtained by brute-force summation over every
 *      assignment of states to the two internal nodes.
 *   2. Time a larger random tree so the kernel changes can be measured rather
 *      than guessed at.
 *
 *   g++ -std=c++20 -O3 -march=native -o bench bench_likelihood.cpp vectorize.cpp -lm
 */

#include "likelihood.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <vector>

/* Build with -DPYML_COUNT_ALLOCS to count heap allocations across the timed
   loop.  The pruner is supposed to run allocation-free once constructed, and
   this is how that is checked rather than assumed. */
#ifdef PYML_COUNT_ALLOCS
static long g_allocs = 0;
void* operator new(std::size_t n) {
    ++g_allocs;
    void* p = std::malloc(n);
    if (p == nullptr) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#endif

using Real = double;

/* ------------------------------------------------------------ reference --- */

/* ((A,C),(B,D)), all branch lengths 0.1, JC69, sequences ACGT / ACGT / ATCG / ATCG.
 * Brute force: sum over the two internal states x, y and the root state r of
 *   pi[r] P[r][x] P[r][y] P[x][A] P[x][C] P[y][B] P[y][D]
 * for each of the four sites, then take logs and add. */
static Real reference_lnL() {
    const Real t = Real{0.1};
    const Real e = std::exp(Real{-4} / Real{3} * t);
    auto P = [&](int i, int j) { return i == j ? Real{0.25} + Real{0.75} * e
                                                : Real{0.25} - Real{0.25} * e; };
    const int seqs[4][4] = {{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 3, 1, 2}, {0, 3, 1, 2}};
    Real total = 0;
    for (int site = 0; site < 4; ++site) {
        const int A = seqs[0][site], B = seqs[1][site];
        const int C = seqs[2][site], D = seqs[3][site];
        Real site_lik = 0;
        for (int r = 0; r < 4; ++r)
            for (int x = 0; x < 4; ++x)
                for (int y = 0; y < 4; ++y)
                    site_lik += Real{0.25} * P(r, x) * P(r, y) *
                                P(x, A) * P(x, C) * P(y, B) * P(y, D);
        total += std::log(site_lik);
    }
    return total;
}

static int check_reference() {
    PYML::SubstMatrix<Real> model;
    std::vector<Real> Q(16), pi(4, Real{0.25});
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            Q[static_cast<std::size_t>(i) * 4 + j] = (i == j) ? Real{-1} : Real{1} / Real{3};
    model.InitWithQPi(Q, pi);

    PYML::BinaryTree<Real> tree(4);
    tree.connect(4, 0, 2);
    tree.connect(5, 1, 3);
    tree.connect(6, 4, 5);
    for (int i = 0; i < tree.NODE_COUNT; ++i)
        tree.Nodes[static_cast<std::size_t>(i)].branch_length = Real{0.1};

    PYML::LikelihoodCalculator<Real>::Alignment aln = {
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 3, 1, 2}, {0, 3, 1, 2}};

    PYML::LikelihoodCalculator<Real> calc(tree, model, aln);
    const Real got = calc.LogLikPruner();
    const Real want = reference_lnL();
    const Real rel = std::abs(got - want) / std::abs(want);
    std::printf("4-taxon JC69  computed %.10f  reference %.10f  rel %.3e\n",
                got, want, rel);
    if (rel > 1e-12) { std::printf("  MISMATCH\n"); return 1; }
    return 0;
}

/* -------------------------------------------------------------- timing --- */

static unsigned rng_state = 20260928u;
static Real rnd() {
    rng_state = rng_state * 1103515245u + 12345u;
    return static_cast<Real>((rng_state >> 8) % 100000) / 100000.0;
}

static PYML::BinaryTree<Real> random_tree(int ntaxa) {
    PYML::BinaryTree<Real> tree(ntaxa);
    std::vector<int> pool(static_cast<std::size_t>(ntaxa));
    for (int i = 0; i < ntaxa; ++i) pool[static_cast<std::size_t>(i)] = i;
    int n = ntaxa, next = ntaxa;
    while (n > 1) {
        const int ia = static_cast<int>(rnd() * static_cast<Real>(n)) % n;
        const int a = pool[static_cast<std::size_t>(ia)];
        pool[static_cast<std::size_t>(ia)] = pool[static_cast<std::size_t>(--n)];
        const int ib = static_cast<int>(rnd() * static_cast<Real>(n)) % n;
        const int b = pool[static_cast<std::size_t>(ib)];
        pool[static_cast<std::size_t>(ib)] = pool[static_cast<std::size_t>(--n)];
        tree.connect(next, a, b);
        tree.Nodes[static_cast<std::size_t>(a)].branch_length = 0.02 + 0.3 * rnd();
        tree.Nodes[static_cast<std::size_t>(b)].branch_length = 0.02 + 0.3 * rnd();
        pool[static_cast<std::size_t>(n++)] = next;
        ++next;
    }
    return tree;
}

int main(int argc, char** argv) {
    if (check_reference() != 0) return 1;

    const int ntaxa = argc > 1 ? std::atoi(argv[1]) : 32;
    const int nsites = argc > 2 ? std::atoi(argv[2]) : 5000;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 50;

    PYML::SubstMatrix<Real> model;
    std::vector<Real> Q(16), pi(4, Real{0.25});
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            Q[static_cast<std::size_t>(i) * 4 + j] = (i == j) ? Real{-1} : Real{1} / Real{3};
    model.InitWithQPi(Q, pi);

    auto tree = random_tree(ntaxa);
    PYML::LikelihoodCalculator<Real>::Alignment aln(
        static_cast<std::size_t>(ntaxa),
        std::vector<int>(static_cast<std::size_t>(nsites)));
    for (auto& row : aln)
        for (auto& s : row) s = static_cast<int>(rnd() * 4.0) % 4;

    PYML::LikelihoodCalculator<Real> calc(tree, model, aln);
    const Real lnL = calc.LogLikPruner();

#ifdef PYML_COUNT_ALLOCS
    g_allocs = 0;
#endif
    const auto t0 = std::chrono::steady_clock::now();
    Real sink = 0;
    for (int r = 0; r < reps; ++r) sink += calc.LogLikPruner();
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
#ifdef PYML_COUNT_ALLOCS
    const long allocs = g_allocs;
#endif

    std::printf("taxa %3d  sites %5d  lnL %.3f  %8.3f ms/eval  (%d reps)",
                ntaxa, nsites, lnL, ms, reps);
#ifdef PYML_COUNT_ALLOCS
    std::printf("  %ld allocations / %d evals = %.1f per eval",
                allocs, reps, static_cast<double>(allocs) / reps);
#endif
    std::printf("\n");
    (void)sink;
    return 0;
}
