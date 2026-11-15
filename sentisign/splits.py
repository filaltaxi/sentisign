import numpy as np


def signer_independent_split(signers, val_frac=0.15, test_frac=0.2, seed=0):
    """Disjoint signers across train/val/test. Raises if overlap."""
    u = np.unique(signers); r = np.random.default_rng(seed); r.shuffle(u)
    nt = max(1, int(round(len(u) * test_frac))); nv = max(1, int(round(len(u) * val_frac)))
    te, va, tr = set(u[:nt]), set(u[nt:nt+nv]), set(u[nt+nv:])
    assert not (te & va or te & tr or va & tr)
    m = lambda S: np.where(np.isin(signers, list(S)))[0]
    return m(tr), m(va), m(te)
