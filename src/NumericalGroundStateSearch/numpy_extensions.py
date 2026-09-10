import numpy as np

def svd_trunc(M, chi_max, tol=1e-15):
    u,s,vh = np.linalg.svd(M, full_matrices=False)
    chi_tol = np.sum(s>tol)
    idx = np.argsort(s)
    idx = idx[::-1]
    m = min(chi_max, chi_tol)
    u = u[:,idx]
    u = u[:,:m]
    s = s[idx]
    s = s[:m]
    vh = vh[idx,:]
    vh = vh[:m,:]
    return u,s,vh

def gs_h(H):
    ew, ev = np.linalg.eigh(H)
    idx = np.argsort(ew)
    ew = ew[idx]
    ev = ev[idx]
    return ew[0], ev[:, 0]