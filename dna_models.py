"""DNA substitution model factories returning configured QMatrix instances.
These functions are intentionally small and accept the `QMatrix` class from `pyml` to avoid import cycles.
"""
import numpy as np


def jc69(QMatrix):
    q = QMatrix(4)
    q.set_attributes(['A', 'C', 'G', 'T'])
    # use symbolic -1 on diagonal consistent with earlier code style
    R = np.ones((4, 4))
    np.fill_diagonal(R, -1)
    # off-diagonals equal
    R = R * 0.25
    np.fill_diagonal(R, -1)
    q.Rmatrix = R
    q.freqs = np.array([0.25, 0.25, 0.25, 0.25])
    return q


def k2p(QMatrix, kappa=2.0):
    q = QMatrix(4)
    q.set_attributes(['A', 'C', 'G', 'T'])
    # transitions: A<->G (0-2), C<->T (1-3)
    R = np.ones((4, 4))
    R *= 1.0  # transversions baseline
    # mark transitions
    R[0, 2] = kappa
    R[2, 0] = kappa
    R[1, 3] = kappa
    R[3, 1] = kappa
    np.fill_diagonal(R, -1)
    q.Rmatrix = R
    q.freqs = np.array([0.25, 0.25, 0.25, 0.25])
    return q


def hky(QMatrix, kappa=2.0, freqs=None):
    q = QMatrix(4)
    q.set_attributes(['A', 'C', 'G', 'T'])
    if freqs is None:
        freqs = np.array([0.25, 0.25, 0.25, 0.25])
    q.freqs = np.asarray(freqs, dtype=float)
    # construct exchangeability R (symmetric) where transitions scaled by kappa
    R = np.ones((4, 4))
    # transitions
    R[0, 2] = kappa
    R[2, 0] = kappa
    R[1, 3] = kappa
    R[3, 1] = kappa
    np.fill_diagonal(R, -1)
    q.Rmatrix = R
    return q


def gtr(QMatrix, rates=None, freqs=None):
    # rates: dictionary or 4x4 matrix of relative exchangeabilities (symmetric)
    q = QMatrix(4)
    q.set_attributes(['A', 'C', 'G', 'T'])
    if freqs is None:
        freqs = np.array([0.25, 0.25, 0.25, 0.25])
    q.freqs = np.asarray(freqs, dtype=float)
    if rates is None:
        R = np.ones((4, 4))
    else:
        rates = np.asarray(rates)
        if rates.shape == (4, 4):
            R = rates.copy()
        else:
            raise ValueError('rates must be 4x4 matrix if provided')
    np.fill_diagonal(R, -1)
    q.Rmatrix = R
    return q
