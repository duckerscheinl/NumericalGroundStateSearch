import numpy as np
from .mps_utils import right_contraction, left_contraction, eff_1s_opm_sc
from ..svd_utils import svd_trunc
from .mps import random_MPS, MPS


# This implementation has to be changed, if one wants to find the groundstate 
# of a non-hermitian operator via DMRG.
def groundstate_dense(H):
    ew, ev = np.linalg.eigh(H)
    idx = np.argsort(ew)
    ev = ev[:,idx]
    ew = ew[idx]
    return ew[0], ev[:,0]


def lowest_energy_subspace(H):
    ew, ev = np.linalg.eigh(H)
    idx = np.argsort(ew)
    ev = ev[:,idx]
    ew = ew[idx]
    idx_gs = (ew == ew[0])
    gsp_dim = np.sum(idx_gs)    # gsp is short for groundspace, which is short for lowest energy subspace.
    return ew[0], ev[:,0:gsp_dim], gsp_dim


def next_rBlock(rB, Vi, Vip1, O1, O2):
    O1part = right_contraction([Vi], O1)
    O2part = right_contraction([Vi,Vip1],O2) 
    rB = np.tensordot(rB, Vi.conj(), [1,2]) # [vi+1 vi+1*] [vi* pi* vi+1*] -> [vi+1 vi* pi*] 
    rB = np.tensordot(Vi, rB, [[1,2],[2,0]]) # [vi pi vi+1] [vi+1 vi* pi*] -> [vi vi*]
    rB += O1part
    rB += O2part
    return rB


def rBlocks(phi, O1, O2):
    N = phi.N
    rBs = list()
    rB = np.ones((1,1), dtype=np.complex128)
    rBs.append(rB)
    Vi = phi.get_V(N-1)
    rB = right_contraction([Vi], O1)
    rBs.append(rB)
    for i in range(N-2,0,-1):
        Vi = phi.get_V(i)
        Vip1 = phi.get_V(i+1)
        rB = next_rBlock(rB=rB, Vi=Vi, Vip1=Vip1, O1=O1, O2=O2)
        rBs.append(rB)

    rBs.reverse()
    return rBs


def next_lBlock(lB, Ui, Uim1, O1, O2):
    O1part = left_contraction([Ui], O1)
    O2part = left_contraction([Uim1,Ui], O2)
    lB = np.tensordot(Ui, lB, [0,0]) # [vi pi vi+1] [vi vi*] -> [pi vi+1 vi*]
    lB = np.tensordot(lB, Ui.conj(), [[0,2],[1,0]]) # [pi vi+1 vi*] [vi* pi* vi+1*] -> [vi+1 vi+1*]
    lB += O1part + O2part
    return lB


def lBlocks(phi, O1, O2):
    N = phi.N
    lBs = list()
    lBs.append(np.ones((1,1), dtype=np.complex128))
    U0 = phi.get_U(0)
    lB = left_contraction([U0], O1)
    lBs.append(lB)
    for i in range(1,N-1):
        Ui = phi.get_U(i)
        Uim1 = phi.get_U(i-1)
        lB = next_lBlock(lB=lB, Ui=Ui, Uim1=Uim1, O1=O1, O2=O2)
        lBs.append(lB)

    return lBs


# ATTENTION:    In the edge cases the energy is incorrect, unless
#               the non-existing block and isometry are chosen to be zero.
def ssHeff(n, d_vi, d_vip1, Uim1, Vip1, lB, rB, O1, O2):
    Heff = np.zeros((n,n), dtype=np.complex128)
    Heff += np.kron(np.kron(np.identity(d_vi),O1),np.identity(d_vip1))
    rO2 = eff_1s_opm_sc(0, op=O2, Us=[], Vs=[Vip1])
    Heff += np.kron(np.identity(d_vi),rO2)
    lO2 = eff_1s_opm_sc(1, op=O2, Us=[Uim1], Vs=[])
    Heff += np.kron(lO2,np.identity(d_vip1))
    Heff += np.kron(lB.T, np.kron(np.identity(2), np.identity(d_vip1)))
    Heff += np.kron(np.kron(np.identity(d_vi),np.identity(2)), rB.T)
    return Heff


def dsHeff(n, d_vl, d_vr, U, V, lB, rB, O1, O2):
    I_vl = np.identity(d_vl)
    I_vr = np.identity(d_vr)
    I_2by2 = np.identity(2)
    Heff = np.zeros((n,n), dtype=np.complex128)
    Heff += np.kron(np.kron(I_vl,O1),np.kron(I_2by2,I_vr))
    Heff += np.kron(np.kron(I_vl,I_2by2),np.kron(O1,I_vr))
    Heff += np.kron(np.kron(I_vl,O2.reshape((4,4))),I_vr)
    lO2 = eff_1s_opm_sc(1, op=O2, Us=[U], Vs=[])
    Heff += np.kron(np.kron(lO2,I_2by2),I_vr)
    rO2 = eff_1s_opm_sc(0, op=O2, Us=[], Vs=[V])
    Heff += np.kron(I_vl,np.kron(I_2by2,rO2))
    Heff += np.kron(np.kron(lB.T, I_2by2),np.kron(I_2by2,I_vr))
    Heff += np.kron(np.kron(I_vl,I_2by2),np.kron(I_2by2, rB.T))
    return Heff


def ds_ltr_update(i, phi, ev0, d_vi, d_vip2, m, N):
    m_eff = np.min([2*d_vi,m+i,m-i+N-2, 2*d_vip2])
    theta_m = np.reshape(ev0, (d_vi*2, 2*d_vip2))
    U,S,V = svd_trunc(theta_m, chi_max=m_eff)
    Ti = np.reshape(U@np.diag(S), (d_vi, 2, m_eff))
    Vip1 = np.reshape(V, (m_eff, 2, d_vip2))
    phi.update_T(i,T=Ti)
    phi.update_V(i+1,V=Vip1)
    Tip1 = np.reshape(np.diag(S)@V, (m_eff, 2, d_vip2))
    Ui = np.reshape(U, (d_vi, 2, m_eff))
    phi.update_T(i+1,T=Tip1)
    phi.update_U(i,U=Ui)
    return Ui


def ds_rtl_update(i, phi, ev0, d_vim1, d_vip1, m, N):
    m_eff = np.min([2*d_vim1,m+i,m-i+N-2, 2*d_vip1])
    theta_m = np.reshape(ev0, (d_vim1*2, 2*d_vip1))
    U,S,V = svd_trunc(theta_m, chi_max=m_eff)
    Ti = np.reshape(np.diag(S)@V, (m_eff, 2, d_vip1))
    Uim1 = np.reshape(U, (d_vim1, 2, m_eff))
    phi.update_T(i,T=Ti)
    phi.update_U(i-1,U=Uim1)
    Tim1 = np.reshape(U@np.diag(S), (d_vim1, 2, m_eff))
    Vi = np.reshape(V, (m_eff, 2, d_vip1))
    phi.update_T(i-1,T=Tim1)
    phi.update_V(i,V=Vi)
    return Vi


def ss_ltr_sweep(phi, O1, O2, rBs):    
    N = phi.N
    E0 = 0
    lBs = [np.zeros((1,1), dtype=np.complex128)]

    Ti = phi.get_T(0)
    d_vi = Ti.shape[0]
    d_vip1 = Ti.shape[2]
    n = d_vi*2*d_vip1
    Vip1 = phi.get_V(1)
    Uim1 = np.zeros((1,2,1), dtype=np.complex128)

    Heff = ssHeff(n=n,d_vi=d_vi,d_vip1=d_vip1,Uim1=Uim1,Vip1=Vip1,lB=lBs[0],rB=rBs[0],O1=O1,O2=O2)
    E0, ev0 = groundstate_dense(Heff)

    Ti = np.reshape(ev0, (d_vi, 2, d_vip1))
    phi.update_T(0,T=Ti)
    U,S,V = np.linalg.svd(np.reshape(Ti, (d_vi*2,d_vip1)), full_matrices=False)
    Ui = np.reshape(U, (d_vi, 2, d_vip1))
    phi.update_U(0,U=Ui)
    Tip1 = np.tensordot(np.diag(S)@V,Vip1, [1,0])
    phi.update_T(1,T=Tip1)
    lB = np.tensordot(Ui, O1, [1,1]) # [vi pi vi+1] [i i*] -> [vi vi+1 i]
    lB = np.tensordot(lB, Ui.conj(), [[0,2],[0,1]]) # [vi vi+1 i] [vi* i* vi+1*] -> [vi+1 vi+1*]
    lBs.append(lB)

    for i in range(1,N-1):
        Ti = phi.get_T(i)
        d_vi = Ti.shape[0]
        d_vip1 = Ti.shape[2]
        n = d_vi*2*d_vip1
        Vip1 = phi.get_V(i+1)
        Uim1 = phi.get_U(i-1)

        Heff = ssHeff(n=n,d_vi=d_vi,d_vip1=d_vip1,Uim1=Uim1,Vip1=Vip1,lB=lBs[i],rB=rBs[i],O1=O1,O2=O2)
        E0, ev0 = groundstate_dense(Heff)
        Ti = np.reshape(ev0, (d_vi, 2, d_vip1))
        phi.update_T(i,T=Ti)
        U,S,V = np.linalg.svd(np.reshape(Ti, (d_vi*2,d_vip1)), full_matrices=False)
        Ui = np.reshape(U, (d_vi, 2, d_vip1))
        phi.update_U(i,U=Ui)
        Tip1 = np.tensordot(np.diag(S)@V,Vip1, [1,0])
        phi.update_T(i+1,T=Tip1)
        lB = next_lBlock(lB=lB,Ui=Ui,Uim1=Uim1,O1=O1,O2=O2)
        lBs.append(lB)

    return E0, lBs


def ds_ltr_sweep(phi, m, O1, O2, rBs):
    N = phi.N
    E0 = 0
    lBs = [np.zeros((1,1), dtype=np.complex128)]

    Ti = phi.get_T(0)
    d_vi = Ti.shape[0]
    Vip2 = phi.get_V(2)
    d_vip2 = Vip2.shape[0]
    n = d_vi*2*2*d_vip2 
    Uim1 = np.zeros((1,2,1), dtype=np.complex128)

    Heff = dsHeff(n=n, d_vl=d_vi, d_vr=d_vip2, U=Uim1, V=Vip2, lB=lBs[0], rB=rBs[1], O1=O1, O2=O2)
    E0, ev0 = groundstate_dense(Heff)

    Ui = ds_ltr_update(i=0, phi=phi, ev0=ev0, d_vi=d_vi, d_vip2=d_vip2, m=m, N=N)
    lB = np.tensordot(Ui, O1, [1,1]) # [vi pi vi+1] [i i*] -> [vi vi+1 i]
    lB = np.tensordot(lB, Ui.conj(), [[0,2],[0,1]]) # [vi vi+1 i] [vi* i* vi+1*] -> [vi+1 vi+1*]
    lBs.append(lB)

    for i in range(1,N-2):
        Ti = phi.get_T(i)
        d_vi = Ti.shape[0]
        Vip2 = phi.get_V(i+2)
        d_vip2 = Vip2.shape[0]
        n = d_vi*2*2*d_vip2 
        Uim1 = phi.get_U(i-1)

        Heff = dsHeff(n=n, d_vl=d_vi, d_vr=d_vip2, U=Uim1, V=Vip2, lB=lBs[i], rB=rBs[i+1], O1=O1, O2=O2)
        e0, ev0 = groundstate_dense(Heff)
        if i == N//2:
            E0=e0

        Ui = ds_ltr_update(i=i, phi=phi, ev0=ev0, d_vi=d_vi, d_vip2=d_vip2, m=m, N=N)
        lB = next_lBlock(lB=lB, Ui=Ui, Uim1=Uim1, O1=O1, O2=O2)
        lBs.append(lB)

    Ti = phi.get_T(N-2)
    d_vi = Ti.shape[0]
    Vip2 = np.zeros((1,2,1))
    d_vip2 = Vip2.shape[0]
    n = d_vi*2*2*d_vip2 
    Uim1 = phi.get_U(N-3)

    Heff = dsHeff(n=n, d_vl=d_vi, d_vr=d_vip2, U=Uim1, V=Vip2, lB=lBs[N-2], rB=rBs[N-1], O1=O1, O2=O2)
    e0, ev0 = groundstate_dense(Heff)

    Ui = ds_ltr_update(i=N-2, phi=phi, ev0=ev0, d_vi=d_vi, d_vip2=d_vip2, m=m, N=N)
    # This left block is not needed. But computation is no problem.
    lB = next_lBlock(lB=lB, Ui=Ui, Uim1=Uim1, O1=O1, O2=O2)
    lBs.append(lB)

    return E0, lBs


def ss_rtl_sweep(phi, O1, O2, lBs):    
    N = phi.N
    E0 = 0
    rBs = list()
    rB = np.zeros((1,1), dtype=np.complex128)
    rBs.append(rB)
    
    Ti = phi.get_T(N-1)
    d_vi = Ti.shape[0]
    d_vip1 = Ti.shape[2]
    n = d_vi*2*d_vip1
    Uim1 = phi.get_U(N-2)
    Vip1 = np.zeros((1,2,1), dtype=np.complex128)

    Heff = ssHeff(n=n,d_vi=d_vi,d_vip1=d_vip1,Uim1=Uim1,Vip1=Vip1,lB=lBs[N-1],rB=rB,O1=O1,O2=O2)
    E0, ev0 = groundstate_dense(Heff)

    Ti = np.reshape(ev0, (d_vi, 2, d_vip1))
    phi.update_T(N-1,T=Ti)
    U,S,V = np.linalg.svd(np.reshape(Ti, (d_vi,2*d_vip1)), full_matrices=False)
    Vi = np.reshape(V, (d_vi, 2, d_vip1))
    phi.update_V(N-1,V=Vi)
    Tim1 = np.tensordot(Uim1,U@np.diag(S), [2,0])
    phi.update_T(N-2,T=Tim1)
    rB = np.tensordot(Vi, O1, [1,1]) # [vi pi vi+1] [i i*] -> [vi vi+1 i]
    rB = np.tensordot(rB, Vi.conj(), [[1,2],[2,1]]) # [vi vi+1 i] [vi* pi* vi+1*] -> [vi vi*]
    rBs.append(rB)

    for i in range(N-2,0,-1):
        Ti = phi.get_T(i)
        d_vi = Ti.shape[0]
        d_vip1 = Ti.shape[2]
        n = d_vi*2*d_vip1
        Vip1 = phi.get_V(i+1)
        Uim1 = phi.get_U(i-1)

        Heff = ssHeff(n=n,d_vi=d_vi,d_vip1=d_vip1,Uim1=Uim1,Vip1=Vip1,lB=lBs[i],rB=rB,O1=O1,O2=O2)
        E0, ev0 = groundstate_dense(Heff)

        Ti = np.reshape(ev0, (d_vi, 2, d_vip1))
        phi.update_T(i,T=Ti)
        U,S,V = np.linalg.svd(np.reshape(Ti, (d_vi,2*d_vip1)), full_matrices=False)
        Vi = np.reshape(V, (d_vi, 2, d_vip1))
        phi.update_V(i,V=Vi)
        Tim1 = np.tensordot(Uim1,U@np.diag(S), [2,0])
        phi.update_T(i-1,T=Tim1)
        rB = next_rBlock(rB=rB, Vi=Vi, Vip1=Vip1, O1=O1, O2=O2)
        rBs.append(rB)

    rBs.reverse()

    return E0, rBs


def ds_rtl_sweep(phi, m, O1, O2, lBs):
    N = phi.N
    E0 = 0
    rB = np.zeros((1,1), dtype=np.complex128)
    rBs = [rB]

    Ti = phi.get_T(N-1)
    d_vip1 = Ti.shape[2]
    Vip1 = np.zeros((1,2,1), dtype=np.complex128)
    Uim2 = phi.get_U(N-3)
    d_vim1 = Uim2.shape[2]
    n = d_vim1*2*2*d_vip1

    Heff = dsHeff(n=n, d_vl=d_vim1, d_vr=d_vip1, U=Uim2, V=Vip1, lB=lBs[N-2], rB=rB, O1=O1, O2=O2)
    E0, ev0 = groundstate_dense(Heff)

    Vi = ds_rtl_update(i=N-1, phi=phi, ev0=ev0, d_vim1=d_vim1, d_vip1=d_vip1, m=m, N=N)
    rB = np.tensordot(Vi, O1, [1,1]) # [vi pi vi+1] [i i*] -> [vi vi+1 i]
    rB = np.tensordot(rB, Vi.conj(), [[1,2],[2,1]]) # [vi vi+1 i] [vi* pi* vi+1*] -> [vi vi*]
    rBs.append(rB)

    for i in range(N-2,1,-1):
        Ti = phi.get_T(i)
        d_vip1 = Ti.shape[2]
        Uim2 = phi.get_U(i-2)
        Vip1 = phi.get_V(i+1)
        d_vim1 = Uim2.shape[2]
        n = d_vim1*2*2*d_vip1 

        Heff = dsHeff(n=n, d_vl=d_vim1, d_vr=d_vip1, U=Uim2, V=Vip1, lB=lBs[i-1], rB=rB, O1=O1, O2=O2)
        e0, ev0 = groundstate_dense(Heff)
        if i == N//2:
            E0 = e0

        Vi = ds_rtl_update(i=i, phi=phi, ev0=ev0, d_vim1=d_vim1, d_vip1=d_vip1, m=m, N=N)
        rB = next_rBlock(rB=rB, Vi=Vi, Vip1=Vip1, O1=O1, O2=O2)
        rBs.append(rB)

    Ti = phi.get_T(1)
    d_vip1 = Ti.shape[2]
    Uim2 = np.zeros((1,2,1))
    d_vim1 = Uim2.shape[2]
    n = d_vim1*2*2*d_vip1 
    Vip1 = phi.get_V(2)

    Heff = dsHeff(n=n, d_vl=d_vim1, d_vr=d_vip1, U=Uim2, V=Vip1, lB=lBs[0], rB=rB, O1=O1, O2=O2)
    e0, ev0 = groundstate_dense(Heff)

    Vi = ds_rtl_update(i=1, phi=phi, ev0=ev0, d_vim1=d_vim1, d_vip1=d_vip1, m=m, N=N)
    # This left block is not needed. But computation is no problem.
    rB = next_rBlock(rB=rB,Vi=Vi,Vip1=Vip1,O1=O1,O2=O2)
    rBs.append(rB)

    rBs.reverse()

    return E0, rBs


def dmrg(N, m, O1, O2, method=2, maxiter=50) -> tuple[float, MPS]:
    
    phi, _ = random_MPS(N=N,chi_max=m)
    E0_old = np.inf
    E_diff = np.inf
    rBs = rBlocks(phi=phi, O1=O1, O2=O2)

    iter = 0
    if method == 1:
        while E_diff > 1e-12 and iter < maxiter:

            E0, lBs = ss_ltr_sweep(phi=phi, O1=O1, O2=O2, rBs=rBs)
            E0, rBs = ss_rtl_sweep(phi=phi, O1=O1, O2=O2, lBs=lBs)

            E_diff = np.abs(E0_old-E0)
            E0_old = E0
            iter += 1
    elif method == 2:
        while E_diff > 1e-12 and iter < maxiter:

            E0, lBs = ds_ltr_sweep(phi=phi, m=m, O1=O1, O2=O2, rBs=rBs)
            E0, rBs = ds_rtl_sweep(phi=phi, m=m, O1=O1, O2=O2, lBs=lBs)

            E_diff = np.abs(E0_old-E0)
            E0_old = E0
            iter += 1
    
    if iter == maxiter:
        print("ssDMRG did not converge!")

    return E0, phi