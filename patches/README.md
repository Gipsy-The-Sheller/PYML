# patches/

## paml-mcmctree-veclnl.patch

Adds an optional SIMD likelihood path (`vecLnL`) to PAML's mcmctree, for
measuring pyml's vector kernels against a real scalar implementation.
Applied on top of `abacus-gene/paml` (paml 4.10.10, 29 Jan 2026).

    git clone https://github.com/abacus-gene/paml.git
    cd paml && git apply <this repo>/patches/paml-mcmctree-veclnl.patch

Verified: `git apply` on a clean checkout reproduces the working tree
byte-for-byte, and that checkout compiles.

Touches two files. Both changes are inert unless `vecLnL` is set in the
control file, so an unpatched run and a `vecLnL = 0` run behave identically.

- `src/mcmctree.c` — `vecLnL` option, `ConditionalPNodeVec`, a state-major
  shadow of the conditional-probability buffers, precomputed tip indicator
  blocks, a `vecLnL = 2` self-check mode that runs both paths on the same
  parameters and reports the per-pattern difference, and timers around
  `com.plfun()`.
- `src/treesub.c` — the `lfun()` hook, guarded by `#ifdef MCMCTREE` so
  baseml/codeml are unaffected.

Build (pyml's kernels are C++, so g++ links):

    gcc  -O3 -march=native -w -I<pyml>/cpp -c mcmctree.c
    gcc  -O3 -march=native -w -c tools.c
    g++  -std=c++20 -O3 -march=native -c <pyml>/cpp/vectorize.cpp
    g++  -std=c++20 -O3 -march=native -c <pyml>/cpp/vectorize_c.cpp
    g++  -O3 -march=native -o mcmctree mcmctree.o tools.o vectorize.o vectorize_c.o -lm

See `note.md` in the repository root for the measurements and what they mean.
