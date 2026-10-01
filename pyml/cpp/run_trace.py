#!/usr/bin/env python3
"""Run a chain and write a Tracer-ready parameter file.

The output is a plain tab-separated table, first row a header, first column
the generation number -- the layout MrBayes writes for its .p files, so Tracer
reads it directly.  Columns whose name shares a prefix and a bracketed index
(`pi(A)`, `pi(C)`, ...) are grouped by Tracer into one partition.

    D:/PyMOL/python.exe pyml/cpp/run_trace.py [phylip] [generations] [out_prefix]

Diagnostics printed alongside are only a quick read on whether the run is
usable; Tracer is the tool for looking properly.
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _core as core


def read_phylip(path):
    with open(path) as fh:
        lines = [l.rstrip("\n") for l in fh if l.strip()]
    ntaxa = int(lines[0].split()[0])
    names, seqs = [], []
    for ln in lines[1:ntaxa + 1]:
        parts = ln.split()
        names.append(parts[0])
        seqs.append("".join(parts[1:]))
    return names, seqs


def ess(x):
    """Effective sample size, initial positive sequence estimator."""
    n = len(x)
    if n < 10:
        return float("nan")
    mean = sum(x) / n
    d = [v - mean for v in x]
    var = sum(v * v for v in d) / n
    if var <= 0:
        return float(n)
    total = 0.0
    for lag in range(1, min(n // 4, 2000)):
        c = sum(d[i] * d[i + lag] for i in range(n - lag)) / (n * var)
        if c <= 0.0:
            break
        total += c
    return n / (1.0 + 2.0 * total)


def summarise(label, x):
    n = len(x)
    mean = sum(x) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in x) / max(n - 1, 1))
    e = ess(x)
    half = n // 2
    m1 = sum(x[:half]) / half
    m2 = sum(x[half:]) / (n - half)
    # Standard error from the ESS, not from the raw count: consecutive samples
    # are correlated, and sqrt(n) would flag every normally-mixing chain.
    se = sd * math.sqrt(2.0 / e) if e > 0 else 0.0
    z = abs(m1 - m2) / se if se > 0 else 0.0
    flag = "" if z < 2.0 else "  <- halves disagree"
    print(f"    {label:9s} mean {mean:12.4f}  sd {sd:9.4f}  "
          f"ESS {e:7.1f}  split-half z {z:4.2f}{flag}")
    return e, z


def main():
    argv = sys.argv[1:]
    phy = argv[0] if argv else "E:/Codes/PYML/paml_bench/data/sim16x2000.phy"
    generations = int(argv[1]) if len(argv) > 1 else 500_000
    prefix = argv[2] if len(argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "trace")
    burnin = generations // 5
    sample_every = max(1, generations // 5000)

    names, seqs = read_phylip(phy)
    ntaxa = len(names)
    print(f"data    {phy}   {ntaxa} taxa x {len(seqs[0])} sites")
    print(f"chain   {generations} generations, burn-in {burnin}, "
          f"sample every {sample_every}")

    data = core.DnaData.from_strings(names, seqs)
    state = core.Core(data, core.Tree.star(ntaxa, 0.1), [0.25] * 4)

    # MrBayes' default move set and relative probabilities for a single DNA
    # partition, restricted to the moves that exist here.  Nodeslider is left
    # out rather than approximated by something that is not it.
    moves, weights = [], []
    for mv, w in [(core.Multiplier(1.386, "V"), 20.0),
                  (core.TLMultiplier(1.386, "V"), 3.0),
                  (core.NNI("Tau,V"), 3.0),
                  (core.DirichletFreqs(100.0, "Pi"), 0.5),
                  (core.SliderFreqs(0.2, "Pi"), 0.5)]:
        mv.weight = w
        moves.append(mv)

    # MrBayes' defaults for a DNA analysis, so the two runs are comparable:
    # statefreqpr=dirichlet(1,1,1,1), brlenspr=unconstrained:gammadir(1,0.1,1,1),
    # topologypr=uniform.  The tree-length prior therefore has mean 1/0.1 = 10,
    # which comfortably covers this dataset; an earlier version of this script
    # used rate=10 (mean 0.1) and the data overruled it.
    priors = [core.DirichletPrior([1.0] * 4, "Pi"),
              core.GammaDirPrior(1.0, 0.1, 1.0, "V"),
              core.FlatPrior("Tau")]

    mcmc = core.Mcmc(state, moves, priors, seed=20260928)
    mcmc.enable_trace(sample_every, with_trees=bool(prefix.endswith(".t")),
                      with_branches=True)

    t0 = time.time()
    # Tune during burn-in, then freeze.  The tuner nudges its parameter by a
    # fixed factor whenever a window's rate misses the target, so it keeps
    # oscillating rather than settling; leaving it on during sampling means the
    # proposal distribution is still changing, which is a real non-stationarity
    # and shows up as a split-half disagreement.  MrBayes stops tuning too.
    mcmc.run(burnin)
    print("moves after burn-in tuning (frozen for the sampling phase):")
    for mv in moves:
        tuned = f"  tuning {mv.tuning:.4g}" if mv.target_rate > 0 else ""
        print(f"    {mv.name:20s} acc {mv.acceptance_rate:.3f}  "
              f"target {mv.target_rate:.2f}{tuned}")
    mcmc.auto_tune = False
    mcmc.run(generations - burnin)
    elapsed = time.time() - t0
    print(f"\n        {elapsed:.1f}s  ({1e3 * elapsed / generations:.3f} ms/gen)")
    print("moves over the whole run:")
    for mv in moves:
        tuned = f"  tuning {mv.tuning:.4g}" if mv.target_rate > 0 else ""
        print(f"    {mv.name:20s} {mv.n_proposed:8d} proposed  "
              f"acc {mv.acceptance_rate:.3f}  target {mv.target_rate:.2f}{tuned}")

    gen = list(mcmc.trace_iteration)
    lls = list(mcmc.trace_loglik)
    lps = list(mcmc.trace_logprior)
    lpo = list(mcmc.trace_logposterior)
    tls = list(mcmc.trace_tree_length)
    fr = list(mcmc.trace_freqs)
    keep = [i for i in range(len(gen)) if gen[i] > burnin]
    take = lambda v: [v[i] for i in keep]

    print(f"\ntrace    {len(gen)} samples recorded, {len(keep)} kept")
    print("  posterior:")
    esss = []
    for label, series in [("LnL", take(lls)), ("TL", take(tls))] + \
                         [(f"pi({c})", [fr[i][k] for i in keep])
                          for k, c in enumerate("ACGT")]:
        e, z = summarise(label, series)
        esss.append(e)
    print(f"  lowest ESS {min(esss):.0f}")

    # ------------------------------------------------------------ output ---
    pfile = prefix + ".p"
    brl = mcmc.trace_branch_lengths
    nbranch = len(brl[0]) if brl else 0
    # Branch columns are named by the smallest tip index on one side of the
    # split, which is stable across topology moves -- see BranchLabel in
    # core/tree.hpp.  A bare index would be a moving target, since NNI relinks
    # nodes without renumbering them.
    header = ["Gen", "LnL", "LnPrior", "LnPost", "TL"] + \
             [f"pi({c})" for c in "ACGT"] + \
             [f"brlen{label}." for label in range(1, nbranch + 1)]
    with open(pfile, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for i in keep:
            row = [str(gen[i]), f"{lls[i]:.6f}", f"{lps[i]:.6f}",
                   f"{lpo[i]:.6f}", f"{tls[i]:.8f}"]
            row += [f"{v:.8f}" for v in fr[i]]
            row += [f"{v:.8f}" for v in brl[i]]
            fh.write("\t".join(row) + "\n")
    print(f"\nwrote {pfile}  ({len(keep)} rows x {len(header)} columns)")
    print("  Tracer reads it directly; pi(*) group into one partition,")
    print("  and brlen1..N are stable branches, not node indices")

    if mcmc.trace_newick:
        tfile = prefix + ".t"
        with open(tfile, "w") as fh:
            fh.write("#NEXUS\nbegin trees;\n")
            for i in keep:
                fh.write(f"  tree gen.{gen[i]} = {mcmc.trace_newick[i]}\n")
            fh.write("end;\n")
        print(f"wrote {tfile}")


if __name__ == "__main__":
    main()
