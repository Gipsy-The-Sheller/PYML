#!/usr/bin/env python3
"""Build pyml._core, the pybind11 module, against PyMOL's Python.

The toolchain combination here is narrow and was found by experiment; the
near misses fail in ways that do not point at the cause, so they are recorded:

  * The compiler must be MSYS2's g++ (/d/msys64/mingw64/bin/g++).  The winlibs
    g++ that `which g++` normally resolves to uses a different thread model and
    its modules do not load.
  * MSYS2's bin directory must be prepended to PATH.  Invoking the compiler by
    absolute path is not enough: it needs its own DLLs on PATH or it exits
    with no diagnostic at all.
  * pybind11 comes from MSYS2 (3.0.2, supports Python >= 3.8); the Python
    headers and the import library come from PyMOL; the target is PyMOL's
    CPython 3.10, which is MSVC-built.  A MinGW-built module does load into it,
    which is what makes this work.
  * -static-libgcc -static-libstdc++ keeps the module from needing
    libstdc++-6.dll and friends at import time.

Every path can be overridden by environment variable; the defaults are the
ones on this machine.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # pyml/cpp
PKG = HERE.parent                               # pyml
REPO = PKG.parent                               # repo root

MSYS = Path(os.environ.get("PYML_MSYS", "D:/msys64"))
MINGW = MSYS / "mingw64"
PYMOL = Path(os.environ.get("PYML_PYMOL", "D:/PyMOL"))

# Where GCC puts its temporaries.  The system temp is on C: here, and C: is
# full, which makes cc1plus die with a misleading "out of memory allocating
# 65536 bytes" even though tens of gigabytes are free elsewhere.
TMP = Path(os.environ.get("PYML_BUILD_TMP", str(REPO.parent / "gcctmp")))

SOURCES = [
    HERE / "vectorize.cpp",
    HERE / "vectorize_c.cpp",
    HERE / "bindings.cpp",
]


def find_compiler() -> Path:
    gxx = MINGW / "bin" / "g++.exe"
    if not gxx.exists():
        sys.exit(f"g++ not found at {gxx}; set PYML_MSYS to the MSYS2 root")
    out = subprocess.run([str(gxx), "-dumpmachine"], capture_output=True, text=True)
    if out.returncode != 0 or "mingw32" not in out.stdout:
        sys.exit(f"{gxx} does not look like an MSYS2 MinGW compiler")
    return gxx


def python_tag() -> str:
    """The EXT_SUFFIX, so the .pyd lands on a name this interpreter imports."""
    ver = f"{sys.version_info.major}{sys.version_info.minor}"
    return f".cp{ver}-win_amd64.pyd"


def main() -> int:
    gxx = find_compiler()
    TMP.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PATH"] = str(MINGW / "bin") + os.pathsep + env.get("PATH", "")
    env["TMPDIR"] = str(TMP)
    env["TEMP"] = str(TMP)
    env["TMP"] = str(TMP)

    out = PKG / ("_core" + python_tag())
    cmd = [
        str(gxx),
        "-std=c++20", "-O3", "-march=native", "-shared",
        "-static-libgcc", "-static-libstdc++",
        f"-I{MINGW / 'include'}",                 # pybind11
        f"-I{str(PYMOL / 'include')}",            # Python.h
        f"-I{HERE}",                              # vectorize.hpp, core/
        *[str(s) for s in SOURCES],
        "-o", str(out),
        f"-L{PYMOL / 'libs'}", "-lpython310",
    ]

    print(" ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        return result.returncode
    print(f"\nbuilt {out}  ({out.stat().st_size} bytes)")
    print(f"import it with:  {PYMOL / 'python.exe'} -c \"import pyml._core\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
