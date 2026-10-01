/* ===========================================================================
 *  test_priors.cpp — check the Hastings ratios by sampling the prior
 *
 *  With the likelihood switched off the target is exactly the prior, which is
 *  known in closed form.  A move whose Hastings ratio is wrong does not
 *  preserve that target, so the sampled distribution drifts away from the
 *  analytic one and the error is measurable.  This is a far stronger check
 *  than eye-balling a trace, and it tests each move on its own.
 *
 *  The most useful case is GammaDir on the branch lengths.  Integrating the
 *  Dirichlet part out leaves the total tree length marginally Gamma(shape,
 *  rate) -- the Jacobian of v = L*p cancels exactly -- so the sampled tree
 *  length has a known mean and variance.  Any wrong power of the multiplier
 *  shifts that distribution, which is how the exponent in TLMultiplier is
 *  pinned down rather than guessed.
 *
 *   g++ -std=c++20 -O3 -march=native -I pyml/cpp -o test_priors \
 *       pyml/cpp/test_priors.cpp pyml/cpp/vectorize.cpp
 * =========================================================================== */

#include "core/mcmc.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace pyml::core;

static int failures = 0;

struct Moments { double mean = 0, var = 0, n = 0; };

static double Z(double x) { return x; }

/// Sample `iterations` times, thinning by `thin`, and accumulate the moments
/// of `pick`.  Discards the first half as burn-in.
template <typename F>
static Moments Sample(Mcmc& mcmc, long iterations, long thin, F pick) {
    Moments m;
    double sum = 0, sumsq = 0;
    for (long i = 0; i < iterations; ++i) {
        mcmc.Step();
        if (i < iterations / 2) continue;
        if (i % thin) continue;
        const double v = pick();
        sum += v;
        sumsq += v * v;
        m.n += 1;
    }
    if (m.n > 1) {
        m.mean = sum / m.n;
        m.var = sumsq / m.n - m.mean * m.mean;
    }
    return m;
}

static bool Close(double got, double want, double tol, const char* what) {
    const double rel = std::abs(got - want) / std::max(std::abs(want), 1e-12);
    if (rel > tol) {
        std::printf("    FAIL %-34s got %.6g  want %.6g  rel err %.3f\n",
                    what, got, want, rel);
        ++failures;
        return false;
    }
    std::printf("    ok   %-34s %.6g  (want %.6g)\n", what, got, want);
    return true;
}

static DnaData SmallData(int n_taxa) {
    std::vector<std::string> names, seqs;
    const char* bases = "ACGT";
    unsigned s = 7;
    for (int i = 0; i < n_taxa; ++i) {
        names.push_back("T" + std::to_string(i + 1));
        std::string seq;
        for (int k = 0; k < 20; ++k) {
            s = s * 1103515245u + 12345u;
            seq.push_back(bases[(s >> 16) % 4]);
        }
        seqs.push_back(seq);
    }
    return DnaData::FromStrings(names, seqs);
}

/* ------------------------------------------------ the exponent, settled --- */

static void TestTLMultiplierExponent(int n_taxa, int exponent) {
    const double shape = 2.0, rate = 10.0;          // Gamma(2,10): mean .2, var .02
    DnaData d = SmallData(n_taxa);
    Tree t = MakeStar(n_taxa, 0.05);
    Core core(d, t, {0.25, 0.25, 0.25, 0.25});

    auto mv = std::make_shared<TLMultiplier>(1.386, "V", exponent);
    std::vector<std::shared_ptr<Move>> moves{mv};
    std::vector<std::shared_ptr<Prior>> priors{
        std::make_shared<GammaDirPrior>(shape, rate, 1.0, "V")};
    Mcmc mcmc(core, moves, priors, 12345);
    mcmc.sample_likelihood = false;

    const auto m = Sample(mcmc, 400000, 4, [&] { return mcmc.tree_length(); });
    std::printf("  TLMultiplier exponent=%s%d:  L mean %.5f (want %.5f)  var %.6f (want %.6f)\n",
                exponent < 0 ? "auto(" : "", exponent,
                m.mean, shape / rate, m.var, shape / (rate * rate));
    if (exponent < 0)
        std::printf("     (auto resolves to the branch count k)\n");
}

/* ---------------------------------------------------------- the moves --- */

static void TestMultiplierWithGammaDir(int n_taxa, double exponent_note) {
    (void)exponent_note;
    const double shape = 2.0, rate = 10.0;
    DnaData d = SmallData(n_taxa);
    Core core(d, MakeStar(n_taxa, 0.05), {0.25, 0.25, 0.25, 0.25});

    std::vector<std::shared_ptr<Move>> moves{std::make_shared<Multiplier>(1.386, "V")};
    std::vector<std::shared_ptr<Prior>> priors{
        std::make_shared<GammaDirPrior>(shape, rate, 1.0, "V")};
    Mcmc mcmc(core, moves, priors, 999);
    mcmc.sample_likelihood = false;

    std::printf("  Multiplier(V) + GammaDir(%g,%g)\n", shape, rate);
    const auto m = Sample(mcmc, 2000000, 20, [&] { return mcmc.tree_length(); });
    Close(m.mean, shape / rate, 0.03, "tree length mean");
    Close(m.var, shape / (rate * rate), 0.06, "tree length variance");
    std::printf("    acceptance %.3f\n",
                moves[0]->AcceptanceRate());
}

static void TestDirichletFreqs(double alpha) {
    DnaData d = SmallData(4);
    Core core(d, MakeStar(4, 0.05), {0.25, 0.25, 0.25, 0.25});

    std::vector<std::shared_ptr<Move>> moves{
        std::make_shared<DirichletFreqs>(100.0, "Pi")};
    std::vector<std::shared_ptr<Prior>> priors{
        std::make_shared<DirichletPrior>(std::vector<double>{alpha, alpha, alpha, alpha},
                                         "Pi")};
    Mcmc mcmc(core, moves, priors, 4242);
    mcmc.sample_likelihood = false;

    // Dirichlet(a,a,a,a): mean 1/4, var (a0-a)/(a0^2 (a0+1)) with a0 = 4a
    const double a0 = 4 * alpha;
    const double want_mean = alpha / a0;
    const double want_var = alpha * (a0 - alpha) / (a0 * a0 * (a0 + 1.0));

    std::printf("  Dirichlet(Pi) + Dirichlet(%.0f)\n", alpha);
    for (int i = 0; i < 4; ++i) {
        const auto m = Sample(mcmc, 400000, 4,
                              [&, i] { return mcmc.freqs()[static_cast<std::size_t>(i)]; });
        if (i == 0) {
            Close(m.mean, want_mean, 0.02, "pi_0 mean");
            Close(m.var, want_var, 0.06, "pi_0 variance");
        }
    }
    std::printf("    acceptance %.3f\n", moves[0]->AcceptanceRate());
}

static void TestSliderFreqs(double alpha) {
    DnaData d = SmallData(4);
    Core core(d, MakeStar(4, 0.05), {0.25, 0.25, 0.25, 0.25});

    std::vector<std::shared_ptr<Move>> moves{std::make_shared<SliderFreqs>(0.2, "Pi")};
    std::vector<std::shared_ptr<Prior>> priors{
        std::make_shared<DirichletPrior>(std::vector<double>{alpha, alpha, alpha, alpha},
                                         "Pi")};
    Mcmc mcmc(core, moves, priors, 777);
    mcmc.sample_likelihood = false;

    const double a0 = 4 * alpha;
    const double want_mean = alpha / a0;
    const double want_var = alpha * (a0 - alpha) / (a0 * a0 * (a0 + 1.0));

    std::printf("  Slider(Pi)    + Dirichlet(%.0f)\n", alpha);
    const auto m = Sample(mcmc, 1000000, 10,
                          [&] { return mcmc.freqs()[0]; });
    Close(m.mean, want_mean, 0.02, "pi_0 mean");
    Close(m.var, want_var, 0.06, "pi_0 variance");
    std::printf("    acceptance %.3f\n", moves[0]->AcceptanceRate());
}

static void TestNniMovesTopology() {
    const int n = 5;                                 // 15 unrooted topologies
    DnaData d = SmallData(n);
    Core core(d, MakeStar(n, 0.05), {0.25, 0.25, 0.25, 0.25});

    std::vector<std::shared_ptr<Move>> moves{std::make_shared<NNI>("Tau,V")};
    std::vector<std::shared_ptr<Prior>> priors{std::make_shared<FlatPrior>("Tau")};
    Mcmc mcmc(core, moves, priors, 31337);
    mcmc.sample_likelihood = false;

    std::vector<std::string> names = d.tip_name;
    std::vector<std::string> seen;
    auto record = [&] {
        // canonical form: the newick string, which is enough to tell
        // topologies apart here since the branch lengths differ too
        const auto s = ToNewick(core.tree(), names);
        for (const auto& x : seen)
            if (x == s) return;
        seen.push_back(s);
    };
    record();
    for (int i = 0; i < 20000; ++i) {
        mcmc.Step();
        Validate(core.tree());
        record();
    }
    std::printf("  NNI on %d taxa: %zu distinct trees visited, acceptance %.3f\n",
                n, seen.size(), moves[0]->AcceptanceRate());
    if (seen.size() < 5) {
        std::printf("    FAIL the chain is not moving through topologies\n");
        ++failures;
    }
}

int main() {
    std::printf("=== prior recovery: does each move preserve its target? ===\n\n");

    std::printf("[TLMultiplier exponent]\n");
    for (int e : {-1, 1, 3, 5})
        TestTLMultiplierExponent(4, e);
    std::printf("\n");

    std::printf("[moves]\n");
    TestMultiplierWithGammaDir(4, 0);
    TestDirichletFreqs(1.0);
    TestSliderFreqs(1.0);
    TestNniMovesTopology();

    std::printf("\n%s\n", failures ? "FAILURES" : "all prior checks passed");
    return failures != 0;
}
