import numpy as np

def svd_trunc(M, chi_max, tol=1e-15):
    U,S,Vh = np.linalg.svd(M, full_matrices=False)
    chi_tol = np.sum(S>tol)
    idx = np.argsort(S)
    idx = idx[::-1]
    m = min(chi_max, chi_tol)
    U = U[:,idx]
    U = U[:,:m]
    S = S[idx]
    S = S[:m]
    Vh = Vh[idx,:]
    Vh = Vh[:m,:]
    return U,S,Vh