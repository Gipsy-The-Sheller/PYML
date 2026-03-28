"""Amino-acid model factories.
Provide simple built-in models (uniform) and placeholders for empirical matrices.
"""
import numpy as np


def uniform_aa(QMatrix):
    """Return a simple equal-rates 20-state model with uniform frequencies."""
    q = QMatrix(20)
    aa_letters = [
        'A','R','N','D','C','Q','E','G','H','I',
        'L','K','M','F','P','S','T','W','Y','V'
    ]
    q.set_attributes(aa_letters)
    R = np.ones((20, 20))
    np.fill_diagonal(R, -1)
    q.Rmatrix = R
    q.freqs = np.full(20, 1.0/20.0)
    return q


def empirical_placeholder(QMatrix, name):
    """Placeholder loader for empirical AA models (JTT/WAG/LG).
    If you have empirical exchangeabilities/frequencies available (from IQ-TREE/PLL source),
    pass them as a 20x20 matrix and a frequency vector to `graft_empirical` instead.
    """
    raise NotImplementedError('Empirical AA models not embedded. Use graft_empirical or provide matrix data.')


def graft_empirical(QMatrix, Rmatrix20, freqs20, aa_order=None):
    """Create a QMatrix from provided 20x20 exchangeabilities and frequencies.
    - Rmatrix20: 20x20 symmetric matrix of relative rates (diagonal ignored)
    - freqs20: length-20 stationary frequencies
    - aa_order: optional list of 20 amino-acid labels matching rows/cols
    """
    q = QMatrix(20)
    if aa_order is not None:
        q.set_attributes(aa_order)
    else:
        q.set_attributes([
            'A','R','N','D','C','Q','E','G','H','I',
            'L','K','M','F','P','S','T','W','Y','V'
        ])
    R = np.asarray(Rmatrix20, dtype=float)
    np.fill_diagonal(R, -1)
    q.Rmatrix = R
    q.freqs = np.asarray(freqs20, dtype=float)
    return q


def poisson(QMatrix, freqs=None):
    """F81-like Poisson model for amino acids: equal exchangeabilities, optional freqs."""
    q = QMatrix(20)
    aa_order = [
        'A','R','N','D','C','Q','E','G','H','I',
        'L','K','M','F','P','S','T','W','Y','V'
    ]
    q.set_attributes(aa_order)
    R = np.ones((20, 20), dtype=float)
    np.fill_diagonal(R, -1)
    q.Rmatrix = R
    if freqs is None:
        q.freqs = np.full(20, 1.0/20.0)
    else:
        q.freqs = np.asarray(freqs, dtype=float)
    return q


def _try_load_empirical(name):
    """Try loading empirical matrices from `aa_matrices/{name}_R.npy` and `{name}_freqs.npy`.
    Returns (R, freqs) or (None, None) if not found.
    """
    # Prefer importing a Python module under aa_matrices (hardcoded tables),
    # fall back to loading .npy files if present.
    try:
        mod = __import__(f'aa_matrices.{name}', fromlist=['*'])
        # module may export <NAME>_R and <NAME>_freqs or generic names
        R = getattr(mod, f'{name.upper()}_R', None)
        freqs = getattr(mod, f'{name.upper()}_freqs', None)
        if R is not None and freqs is not None:
            return np.asarray(R, dtype=float), np.asarray(freqs, dtype=float)
    except Exception:
        pass

    import os
    base = os.path.dirname(__file__)
    rpath = os.path.join(base, 'aa_matrices', f'{name}_R.npy')
    fpath = os.path.join(base, 'aa_matrices', f'{name}_freqs.npy')
    if os.path.exists(rpath) and os.path.exists(fpath):
        R = np.load(rpath)
        freqs = np.load(fpath)
        return R, freqs
    return None, None


def jtt(QMatrix, freqs=None):
    """Factory for JTT empirical model. If local empirical files exist, use them;
    otherwise fall back to uniform Poisson-like model or provided `freqs`.
    To embed true JTT parameters, place `jtt_R.npy` and `jtt_freqs.npy` under `aa_matrices/`.
    """
    R, fq = _try_load_empirical('jtt')
    if R is not None:
        if freqs is not None:
            fq = np.asarray(freqs, dtype=float)
        return graft_empirical(QMatrix, R, fq)
    # fallback
    if freqs is None:
        return uniform_aa(QMatrix)
    return poisson(QMatrix, freqs=freqs)


def wag(QMatrix, freqs=None):
    """Factory for WAG empirical model. Uses local files if present, else falls back."""
    R, fq = _try_load_empirical('wag')
    if R is not None:
        if freqs is not None:
            fq = np.asarray(freqs, dtype=float)
        return graft_empirical(QMatrix, R, fq)
    if freqs is None:
        return uniform_aa(QMatrix)
    return poisson(QMatrix, freqs=freqs)
