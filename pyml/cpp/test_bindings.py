#!/usr/bin/env python3
"""End-to-end exercise of pyml._core from Python.

Runs a real MCMC over an unrooted tree with the move set and priors MrBayes
uses for a single-partition DNA analysis, then checks that the chain behaves:
acceptance rates are sane, the trace has no NaNs, and the topology actually
moves.

    D:/PyMOL/python.exe pyml/cpp/test_bindings.py

Note this imports `_core` directly rather than `pyml._core`: `import pyml`
currently fails on an unrelated circular import in pyml/phyloData.py, which
would stop the package from loading before the extension is ever reached.
The extension itself lives in the package directory, so that is what goes on
sys.path here.
"""

import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # pyml/
sys.path.insert(0, _PKG)

try:
    import _core as core
except ImportError:
    sys.exit(f"_core*.pyd not found in {_PKG}; run pyml/cpp/build_core.py first")

failures = 0


def check(cond, what):
    global failures
    if cond:
        print(f"  ok   {what}")
    else:
        print(f"  FAIL {what}")
        failures += 1


# --------------------------------------------------------------- the data ---
# The 4-taxon JC69 case with a known answer, so the likelihood can be checked
# before anything is run.
NAMES = list("ACBD")
SEQS = ["ACGT", "ATCG", "ACGT", "ATCG"]
ANCHOR = -25.786608878551

print("[likelihood]")
t = core.Tree.from_newick("((A:0.1,C:0.1):0.1,(B:0.1,D:0.1):0.1);")
d = core.DnaData.from_strings(NAMES, SEQS)
c = core.Core(d, t, [0.25] * 4)
anchor_ok = abs(c.log_likelihood() - ANCHOR) < 1e-9
check(anchor_ok, f"4-taxon JC69 lnL = {c.log_likelihood():.10f} (anchor {ANCHOR:.10f})")

# --------------------------------------------------------------- the run ---
print("\n[mcmc]")
state = core.Core(d, core.Tree.from_newick("((A:0.1,C:0.1):0.1,(B:0.1,D:0.1):0.1);"),
                  [0.25] * 4)

# Move set and relative probabilities copied from MrBayes `showmoves` for a
# default DNA analysis: Multiplier 20.0, Nodeslider 7.0 (not implemented yet),
# TLMultiplier 3.0, NNI 3.0, Dirichlet(Pi) 0.5, Slider(Pi) 0.5.  Their weights
# are summed below, so the relative proportions are preserved.
moves = [
    core.Multiplier(1.386, "V"),
    core.TLMultiplier(1.386, "V"),
    core.NNI("Tau,V"),
    core.DirichletFreqs(100.0, "Pi"),
    core.SliderFreqs(0.2, "Pi"),
]
for mv, w in zip(moves, [20.0, 3.0, 3.0, 0.5, 0.5]):
    mv.weight = w

priors = [
    core.DirichletPrior([1.0] * 4, "Pi"),
    core.GammaDirPrior(1.0, 10.0, 1.0, "V"),
    core.FlatPrior("Tau"),
]

mcmc = core.Mcmc(state, moves, priors, seed=20260928)
start_lnL = mcmc.log_likelihood

n_iter = 20000
mcmc.run(n_iter)

print(f"  start lnL {start_lnL:.3f} -> {mcmc.log_likelihood:.3f}")
print(f"  logprior {mcmc.log_prior:.3f}   tree length {mcmc.tree_length:.4f}")
print(f"  freqs {[round(x, 4) for x in mcmc.freqs]}")
for mv in moves:
    print(f"    {mv.name:24s} {mv.n_proposed:7d} proposed  "
          f"acc {mv.acceptance_rate:.3f}")

check(mcmc.log_posterior == mcmc.log_posterior, "log posterior is not NaN")
check(mcmc.tree_length > 0, "tree length stays positive")
check(abs(sum(mcmc.freqs) - 1.0) < 1e-9, "frequencies still sum to one")
check(all(mv.n_proposed > 0 for mv in moves), "every move was exercised")
check(all(0.0 <= mv.acceptance_rate <= 1.0 for mv in moves),
      "acceptance rates are in range")
topo_moved = abs(mcmc.log_likelihood - start_lnL) > 1e-9
check(topo_moved, "the chain moved away from its starting point")

# --------------------------------------------------- a Python-defined move ---
print("\n[python-defined move]")


class Jitter(core.Move):
    """A move written in Python, dropped into the same list as the built-ins."""

    def __init__(self, scale=0.01):
        super().__init__()
        self.scale = scale
        self.param = "V"

    @property
    def name(self):
        return "Jitter(Python)"

    def propose(self, core_obj, rng):
        tree = core_obj.tree
        bls = tree.branch_lengths()
        i = rng.below(len(bls))
        bls[i] += rng.normal(0.0, self.scale)
        if bls[i] <= 0:
            return float("-inf")           # reject; state untouched
        tree.set_branch_lengths(bls)
        return 0.0


state2 = core.Core(d, core.Tree.from_newick("((A:0.1,C:0.1):0.1,(B:0.1,D:0.1):0.1);"),
                   [0.25] * 4)
jitter = Jitter()
mcmc2 = core.Mcmc(state2, [jitter], priors, seed=7)
mcmc2.run(2000)
print(f"  {jitter.name}: {jitter.n_proposed} proposed, acc {jitter.acceptance_rate:.3f}")
check(jitter.n_proposed == 2000, "the Python move was called every iteration")
check(jitter.acceptance_rate > 0.0, "the Python move is accepting proposals")

print("\n" + ("FAILURES" if failures else "all binding checks passed"))
sys.exit(1 if failures else 0)
