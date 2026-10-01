/* ===========================================================================
 *  bindings.cpp — pyml._core, the pybind11 module
 *
 *  Exposes the C++ MCMC core so the state, moves and priors can be configured
 *  from Python.  Move and Prior carry trampolines, so a Python subclass can
 *  define its own proposal or density and drop it into the same list as the
 *  built-ins -- that is the counterpart of the propose_func freedom that
 *  pyml.mcmc gives.
 *
 *  Build with pyml/cpp/build_core.py, not by hand: the toolchain combination
 *  that works is specific (MSYS2 g++, MSYS2 pybind11, PyMOL's Python headers
 *  and import library) and getting it wrong fails in confusing ways.
 * =========================================================================== */

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "core/mcmc.hpp"

namespace py = pybind11;
using namespace pyml::core;

/* ------------------------------------------------------------ trampolines --- */

/// The _NAME variants matter: pybind11's override lookup searches for a Python
/// attribute whose name is the one given to py::class_, and that is the
/// lower-case `propose` / `log_prior`.  Using the plain macro makes it look
/// for `Propose`, find nothing, and report "tried to call pure virtual
/// function" at the first step of the run.
class PyMove : public Move {
public:
    using Move::Move;
    double Propose(Core& core, Rng& rng) override {
        PYBIND11_OVERRIDE_PURE_NAME(double, Move, "propose", Propose, core, rng);
    }
};

class PyPrior : public Prior {
public:
    using Prior::Prior;
    [[nodiscard]] double LogPrior(const Core& core) const override {
        PYBIND11_OVERRIDE_PURE_NAME(double, Prior, "log_prior", LogPrior, core);
    }
};

/* ------------------------------------------------------------------ module --- */

PYBIND11_MODULE(_core, m) {
    m.doc() = "pyml's C++ MCMC core: unrooted trees, moves, priors, driver";

    py::class_<Rng>(m, "Rng")
        .def(py::init<std::uint64_t>(), py::arg("seed"))
        .def("uniform", [](Rng& r) { return r.Uniform(); })
        .def("below", &Rng::Below, py::arg("n"))
        .def("normal", &Rng::Normal, py::arg("mean"), py::arg("sd"))
        .def("gamma", &Rng::Gamma, py::arg("shape"), py::arg("scale") = 1.0)
        .def("exponential", &Rng::Exponential, py::arg("rate"))
        .def("log_multiplier", &Rng::LogMultiplier, py::arg("lambda"));

    py::class_<DnaData>(m, "DnaData")
        .def_static("from_strings", &DnaData::FromStrings,
                    py::arg("names"), py::arg("sequences"),
                    py::arg("alphabet") = "ACGT")
        .def_readonly("n_taxa", &DnaData::n_taxa)
        .def_readonly("n_sites", &DnaData::n_sites)
        .def_readonly("n_states", &DnaData::n_states)
        .def_property_readonly("tip_names",
                               [](const DnaData& d) { return d.tip_name; });

    py::class_<Tree>(m, "Tree")
        .def_static("unrooted", &Tree::Unrooted, py::arg("n_taxa"))
        .def_static("star", &MakeStar, py::arg("n_taxa"), py::arg("branch_length"))
        .def_static("from_newick", [](const std::string& s) { return FromNewick(s); },
                    py::arg("newick"))
        .def("to_newick", &ToNewick, py::arg("tip_names"))
        .def("branch_count", [](const Tree& t) { return BranchCount(t); })
        .def("n_nodes", [](const Tree& t) { return t.NODE_COUNT; })
        .def("tree_length", [](const Tree& t) {
            double total = 0.0;
            const int r = Root(t);
            for (int v = 0; v < t.NODE_COUNT; ++v)
                if (v != r) total += At(t, v).branch_length;
            return total;
        })
        .def("branch_lengths", [](const Tree& t) {
            std::vector<double> out;
            const int r = Root(t);
            for (int v = 0; v < t.NODE_COUNT; ++v)
                if (v != r) out.push_back(At(t, v).branch_length);
            return out;
        })
        .def("set_branch_lengths", [](Tree& t, const std::vector<double>& v) {
            const int r = Root(t);
            std::size_t i = 0;
            for (int n = 0; n < t.NODE_COUNT; ++n) {
                if (n == r) continue;
                if (i >= v.size()) throw std::invalid_argument("not enough values");
                At(t, n).branch_length = v[i++];
            }
        }, py::arg("values"))
        .def("validate", &Validate);

    py::class_<Core>(m, "Core")
        .def(py::init<const DnaData&, Tree, std::vector<double>>(),
             py::arg("data"), py::arg("tree"), py::arg("freqs"))
        .def("log_likelihood", &Core::LogLik)
        .def("tree_length", &Core::TreeLength)
        .def_property_readonly("freqs", [](const Core& c) { return c.freqs(); })
        .def("set_freqs", [](Core& c, std::vector<double> pi) {
            if (!c.TrySetFreqs(std::move(pi)))
                throw std::invalid_argument("frequencies rejected (degenerate rate matrix)");
        }, py::arg("freqs"))
        .def_property_readonly("tree", [](Core& c) -> Tree& { return c.tree(); },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("n_states", &Core::n_states);

    /* ------------------------------------------------------------- moves --- */

    py::class_<Move, PyMove, std::shared_ptr<Move>>(m, "Move")
        .def(py::init<>())
        .def_property("weight", [](const Move& x) { return x.weight; },
                      [](Move& x, double w) { x.weight = w; })
        .def_property("param", [](const Move& x) { return x.param; },
                      [](Move& x, const std::string& s) { x.param = s; })
        .def_property_readonly("name", &Move::Name)
        .def_property_readonly("n_proposed", [](const Move& x) { return x.n_proposed; })
        .def_property_readonly("n_accepted", [](const Move& x) { return x.n_accepted; })
        .def_property_readonly("acceptance_rate", &Move::AcceptanceRate)
        .def_property("target_rate", [](const Move& x) { return x.target_rate; },
                      [](Move& x, double v) { x.target_rate = v; },
                      "acceptance rate the auto-tuner aims for; 0 disables tuning")
        .def_property_readonly("window_rate", &Move::WindowRate)
        .def_property_readonly("tuning", &Move::tuning)
        .def("propose", &Move::Propose, py::arg("core"), py::arg("rng"));

    py::class_<Multiplier, Move, std::shared_ptr<Multiplier>>(m, "Multiplier")
        .def(py::init<double, std::string>(),
             py::arg("lambda") = 1.386, py::arg("param") = "V")
        .def_property("lambda", &Multiplier::lambda, &Multiplier::set_lambda);

    py::class_<TLMultiplier, Move, std::shared_ptr<TLMultiplier>>(m, "TLMultiplier")
        .def(py::init<double, std::string, int>(),
             py::arg("lambda") = 1.386, py::arg("param") = "V",
             py::arg("exponent") = -1,
             "exponent = -1 uses the branch count, which is what the prior\n"
             "recovery test in test_priors.cpp measures to be correct")
        .def_property("lambda", &TLMultiplier::lambda, &TLMultiplier::set_lambda)
        .def_property("exponent", &TLMultiplier::exponent, &TLMultiplier::set_exponent);

    py::class_<NNI, Move, std::shared_ptr<NNI>>(m, "NNI")
        .def(py::init<std::string>(), py::arg("param") = "Tau,V");

    py::class_<DirichletFreqs, Move, std::shared_ptr<DirichletFreqs>>(m, "DirichletFreqs")
        .def(py::init<double, std::string>(),
             py::arg("alpha") = 100.0, py::arg("param") = "Pi")
        .def_property("alpha", &DirichletFreqs::alpha, &DirichletFreqs::set_alpha);

    py::class_<SliderFreqs, Move, std::shared_ptr<SliderFreqs>>(m, "SliderFreqs")
        .def(py::init<double, std::string>(),
             py::arg("delta") = 0.2, py::arg("param") = "Pi")
        .def_property("delta", &SliderFreqs::delta, &SliderFreqs::set_delta);

    /* ------------------------------------------------------------ priors --- */

    py::class_<Prior, PyPrior, std::shared_ptr<Prior>>(m, "Prior")
        .def(py::init<>())
        .def_property("param", [](const Prior& x) { return x.param; },
                      [](Prior& x, const std::string& s) { x.param = s; })
        .def_property_readonly("name", &Prior::Name)
        .def("log_prior", &Prior::LogPrior, py::arg("core"));

    py::class_<FlatPrior, Prior, std::shared_ptr<FlatPrior>>(m, "FlatPrior")
        .def(py::init<std::string>(), py::arg("param") = "Tau");

    py::class_<DirichletPrior, Prior, std::shared_ptr<DirichletPrior>>(m, "DirichletPrior")
        .def(py::init<std::vector<double>, std::string>(),
             py::arg("alpha"), py::arg("param") = "Pi")
        .def_property_readonly("alpha", &DirichletPrior::alpha);

    py::class_<GammaDirPrior, Prior, std::shared_ptr<GammaDirPrior>>(m, "GammaDirPrior")
        .def(py::init<double, double, double, std::string>(),
             py::arg("shape") = 1.0, py::arg("rate") = 0.1,
             py::arg("alpha") = 1.0, py::arg("param") = "V")
        .def_property_readonly("shape", &GammaDirPrior::shape)
        .def_property_readonly("rate", &GammaDirPrior::rate)
        .def_property_readonly("alpha", &GammaDirPrior::alpha);

    /* ------------------------------------------------------------- mcmc --- */

    py::class_<Mcmc>(m, "Mcmc")
        .def(py::init<Core&, std::vector<std::shared_ptr<Move>>,
                      std::vector<std::shared_ptr<Prior>>, std::uint64_t>(),
             py::arg("core"), py::arg("moves"), py::arg("priors"), py::arg("seed") = 1,
             py::keep_alive<1, 2>())
        .def_readwrite("sample_likelihood", &Mcmc::sample_likelihood,
                       "set False to sample the prior instead of the posterior")
        .def_readwrite("auto_tune", &Mcmc::auto_tune)
        .def_readwrite("tune_window", &Mcmc::tune_window)
        .def("enable_trace", &Mcmc::EnableTrace,
             py::arg("interval") = 100, py::arg("with_trees") = false,
             py::arg("with_branches") = false)
        .def("discard_trace", &Mcmc::DiscardTrace, py::arg("n"))
        .def_property_readonly("iterations", &Mcmc::iterations)
        .def_property_readonly("tracing", &Mcmc::tracing)
        .def_property_readonly("trace_iteration", &Mcmc::trace_iteration)
        .def_property_readonly("trace_loglik", &Mcmc::trace_loglik)
        .def_property_readonly("trace_logprior", &Mcmc::trace_logprior)
        .def_property_readonly("trace_logposterior", &Mcmc::trace_logposterior)
        .def_property_readonly("trace_tree_length", &Mcmc::trace_tree_length)
        .def_property_readonly("trace_freqs", &Mcmc::trace_freqs)
        .def_property_readonly("trace_branch_lengths", &Mcmc::trace_branch_lengths)
        .def_property_readonly("trace_newick", &Mcmc::trace_newick)
        .def("step", &Mcmc::Step)
        .def("run", &Mcmc::Run, py::arg("iterations"))
        .def("refresh", &Mcmc::Refresh)
        .def_property_readonly("log_likelihood", &Mcmc::log_likelihood)
        .def_property_readonly("log_prior", &Mcmc::log_prior)
        .def_property_readonly("log_posterior", &Mcmc::log_posterior)
        .def_property_readonly("tree_length", &Mcmc::tree_length)
        .def_property_readonly("freqs", &Mcmc::freqs)
        .def_property_readonly("moves", &Mcmc::moves);

    m.attr("kMinFreq") = kMinFreq;
}
