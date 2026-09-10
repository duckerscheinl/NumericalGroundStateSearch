import numpy as np
from ..numpy_extensions import svd_trunc, gs_h
from numpy import kron, eye


def initialize_dmrg_eff(N,chi_max,O_os,O_l,O_r):
    assert N%2 == 0
    HL_dict = dict()
    HL_dict[0] = np.zeros(1)
    HR_dict = dict()
    HR_dict[0] = np.zeros(1)
    bO_l_dict = dict()
    bO_l_dict[0] = np.zeros(1)
    bO_r_dict = dict()
    bO_r_dict[0] = np.zeros(1)
    HL = 0
    HR = 0
    bO_l = 0
    bO_r = 0
    n = 0
    M = 2**n
    while n < N/2:
        HL = kron(HL,eye(2)) + kron(eye(M),O_os) + kron(bO_l,O_r)
        HR = kron(eye(2),HR) + kron(O_os,eye(M)) + kron(O_l,bO_r)
        chi = min(2*M, chi_max)
        H = kron(HL,eye(2*M)) + kron(eye(2*M),HR) + kron(kron(eye(M),O_l),kron(O_r,eye(M)))
        ew, ev = eigh(H,k=1,which="SA")
        V0 = ev[:,0]
        gs_m = V0.reshape(2*M,2*M)
        u, s, vh = svd(gs_m, k=chi, tol=1e-12, which="LM")
        HL = u.T.conj()@HL@u
        HR = vh.conj()@HR@vh.T
        bO_l = u.T.conj()@kron(eye(M),O_l)@u
        bO_r = vh.conj()@kron(O_r,eye(M))@vh.T

        M = s.shape[0]
        n += 1
        HL_dict[n] = HL
        HR_dict[n] = HR
        bO_l_dict[n] = bO_l
        bO_r_dict[n] = bO_r
        
    return HL_dict, HR_dict, bO_l_dict, bO_r_dict


class MatrixDMRG:

    def __init__(self, N, chi_max, O_os, O_l, O_r):
        assert N%2==0
        assert N >= 6
        self.N = N
        self.chi_max = chi_max
        self.O_os = O_os
        self.O_l = O_l
        self.O_r = O_r
        self.HL_dict = dict()
        self.HL_dict[0] = np.zeros(1)
        self.HR_dict = dict() 
        self.HR_dict[0] = np.zeros(1)
        self.bO_l_dict = dict()
        self.bO_l_dict[0] = np.zeros(1)
        self.bO_r_dict = dict()
        self.bO_r_dict[0] = np.zeros(1)

    def init_dmrg(self):
        n = 0
        m = 2**n
        while n < self.N//2:
            H, HLp1, HRp1, _, _ = self.build_hamiltonian(nL=n, nR=n)
            e0, v0 = gs_h(H)
            gs_m = v0.reshape(2*m,2*m)
            chi = min(2*m, self.chi_max)
            u, s, vh = svd_trunc(M=gs_m, chi_max=chi, tol=1e-12)
            HL = u.T.conj()@HLp1@u
            HR = vh.conj()@HRp1@vh.T
            bO_l = u.T.conj()@kron(eye(m),self.O_l)@u
            bO_r = vh.conj()@kron(self.O_r,eye(m))@vh.T
    
            m = s.shape[0]
            n += 1
            self.HL_dict[n] = HL
            self.HR_dict[n] = HR
            self.bO_l_dict[n] = bO_l
            self.bO_r_dict[n] = bO_r

    def build_hamiltonian(self, nL, nR):

        HL = self.HL_dict[nL]
        ML = HL.shape[0]
        bO_l = self.bO_l_dict[nL]
        HR = self.HR_dict[nR]
        MR = HR.shape[0]
        bO_r = self.bO_r_dict[nR]
        
        HLp1 = kron(HL,eye(2)) + kron(bO_l,self.O_r) + kron(eye(ML),self.O_os)
        HRp1 = kron(eye(2),HR) + kron(self.O_l,bO_r) + kron(self.O_os,eye(MR))
        H = kron(HLp1,eye(2*MR)) + kron(eye(2*ML),HRp1) + kron(kron(eye(ML),self.O_l),kron(self.O_r,eye(MR)))

        return H, HLp1, HRp1, ML, MR

    def ltr_sweep(self, nL, nR):
        
        H, HLp1, HRp1, ML, MR = self.build_hamiltonian(nL=nL, nR=nR)
        e0, v0 = gs_h(H)
        gs_m = v0.reshape(2*ML,2*MR)
        chi = min(2*ML, self.chi_max)
        u, s, vh = svd_trunc(gs_m, chi_max=chi, tol=1e-12)
        HL = u.T.conj()@HLp1@u
        bO_l = u.T.conj()@kron(eye(ML),self.O_l)@u
        self.HL_dict[nL+1] = HL
        self.bO_l_dict[nL+1] = bO_l
    
        return e0

    def rtl_sweep(self, nL, nR):

        H, HLp1, HRp1, ML, MR = self.build_hamiltonian(nL=nL, nR=nR)
        e0, v0 = gs_h(H)
        gs_m = v0.reshape(2*ML,2*MR).T
        chi = min(2*MR, self.chi_max)
        u, s, vh = svd_trunc(gs_m, chi_max=chi, tol=1e-12)
        HR = u.T.conj()@HRp1@u
        bO_r = u.T.conj()@kron(self.O_r,eye(MR))@u
        self.HR_dict[nR+1] = HR
        self.bO_r_dict[nR+1] = bO_r

        return e0

    def run(self, maxiter=50, conv=1e-12):

        self.init_dmrg()
        E0 = np.inf
        nL = self.N//2
        nR = self.N-nL-2
        for i in range(maxiter):
            while nL <= self.N-3:
                E0_iter = self.ltr_sweep(nL=nL,nR=nR)
                if nL == self.N//2:
                    E_diff = np.abs(E0-E0_iter)
                    E0 = E0_iter
                    if E_diff < conv:
                        return E0
                nL += 1
                nR -= 1
            while nR <= self.N-3:
                E0_iter = self.rtl_sweep(nL=nL,nR=nR)
                nL -= 1
                nR += 1

        print("DMRG did not converge!")
        return E0



def initialize_dmrg(N,m,O_os,O_l,O_r):
    assert N%2 == 0
    HL_dict = dict()
    HL_dict[0] = np.zeros(1)
    HR_dict = dict()
    HR_dict[0] = np.zeros(1)
    bO_l_dict = dict()
    bO_l_dict[0] = np.zeros(1)
    bO_r_dict = dict()
    bO_r_dict[0] = np.zeros(1)
    HL = 0
    HR = 0
    bO_l = 0
    bO_r = 0
    n = 0
    M = 2**n
    while n < N/2:
        HL = kron(HL,eye(2)) + kron(eye(M),O_os) + kron(bO_l,O_r)
        HR = kron(eye(2),HR) + kron(O_os,eye(M)) + kron(O_l,bO_r)
        if 2*M <= m:
            bO_l = kron(eye(M),O_l)
            bO_r = kron(O_r,eye(M))
            n += 1
            M = 2*M
        else:
            H = kron(HL,eye(2*M)) + kron(eye(2*M),HR) + kron(kron(eye(M),O_l),kron(O_r,eye(M)))
            ew, ev = eigsh(H,k=1,which="SA")
            E0 = ew[0]
            V0 = ev[:,0]
            helper = V0.reshape(2*M,2*M)
            #rho is a numpy array not a sparse matrix.
            rho_Lp1 = helper@helper.T.conj() 
            assert np.allclose(1,np.trace(rho_Lp1))
            assert np.allclose(rho_Lp1,rho_Lp1.T.conj())
            ew, ev = eigsh(rho_Lp1, m, which="LM")
            ew[np.abs(ew) < 1e-14] = 0.0
            assert np.sum(ew>=0) == ew.size
            H = ev.T.conj()@HL@ev
            HL = H
            HR = H
            bO_l = ev.T.conj()@kron(eye(M),O_l)@ev
            bO_r = ev.T.conj()@kron(O_r,eye(M))@ev
            M = m
            n += 1

        HL_dict[n] = HL
        HR_dict[n] = HR
        bO_l_dict[n] = bO_l
        bO_r_dict[n] = bO_r
        
    return HL_dict, HR_dict, bO_l_dict, bO_r_dict


def finite_dmrg(N,m,O_os,O_l,O_r,maxiter=50): 
    assert N%2==0
    assert N >= 6

    HL_dict, HR_dict, bO_l_dict, bO_r_dict = initialize_dmrg(N=N,m=m,O_os=O_os,O_l=O_l,O_r=O_r)
    
    E0_gs = np.inf
    nL = N//2
    nR = N-nL-2
    for i in range(maxiter):
        while nL <= N-3:
            HL = HL_dict[nL]
            ML = HL.shape[0]
            bO_l = bO_l_dict[nL]
            HR = HR_dict[nR]
            MR = HR.shape[0]
            bO_r = bO_r_dict[nR]
            HLp1 = kron(HL,eye(2)) + kron(bO_l,O_r) + kron(eye(ML),O_os)
            HRp1 = kron(eye(2),HR) + kron(O_l,bO_r) + kron(O_os,eye(MR))
            if 2*ML > m:
                H = kron(HLp1,eye(2*MR)) + kron(eye(2*ML),HRp1) + kron(kron(eye(ML),O_l),kron(O_r,eye(MR)))
                ew, ev = eigsh(H,k=1,which="SA")
                E0 = ew[0]
                if nL == N//2:
                    E_diff = np.abs(E0 - E0_gs)
                    E0_gs = E0
                    if E_diff < 1e-12:
                        print("Converged!")
                        return E0_gs

                V0 = ev[:,0]
                helper = V0.reshape(2*ML,2*MR)
                rho_Lp1 = helper@helper.T.conj()
                assert np.allclose(1,np.trace(rho_Lp1))
                assert np.allclose(rho_Lp1,rho_Lp1.T.conj())
                ew, ev = eigsh(rho_Lp1, m, which="LM")
                ew[np.abs(ew) < 1e-14] = 0.0
                assert np.sum(ew>=0) == ew.size
                HL = ev.T.conj()@HLp1@ev
                bO_l = ev.T.conj()@kron(eye(ML),O_l)@ev
            else:
                if nL == N//2:
                    H = kron(HLp1,eye(2*MR)) + kron(eye(2*ML),HRp1) + kron(kron(eye(ML),O_l),kron(O_r,eye(MR)))
                    ew, ev = eigsh(H,k=1,which="SA")
                    E0 = ew[0]
                    E0_gs = E0
                HL = HLp1
                bO_l = kron(eye(ML),O_l)
            nL += 1
            nR -= 1
            HL_dict[nL] = HL
            bO_l_dict[nL] = bO_l

        while nR <= N-3:
            HL = HL_dict[nL]
            ML = HL.shape[0]
            bO_l = bO_l_dict[nL]
            HR = HR_dict[nR]
            MR = HR.shape[0]
            bO_r = bO_r_dict[nR]
            HLp1 = kron(HL,eye(2)) + kron(bO_l,O_r) + kron(eye(ML),O_os)
            HRp1 = kron(eye(2),HR) + kron(O_l,bO_r) + kron(O_os,eye(MR))
            if 2*MR > m:
                H = kron(HLp1,eye(2*MR)) + kron(eye(2*ML),HRp1) + kron(kron(eye(ML),O_l),kron(O_r,eye(MR)))
                ew, ev = eigsh(H,k=1,which="SA")
                E0 = ew[0]
                V0 = ev[:,0]
                helper = V0.reshape(2*ML,2*MR)
                rho_Rp1 = helper.T@helper.conj()
                assert np.allclose(1,np.trace(rho_Rp1))
                assert np.allclose(rho_Rp1,rho_Rp1.T.conj())
                ew, ev = eigsh(rho_Rp1, m, which="LM")
                ew[np.abs(ew) < 1e-14] = 0.0
                assert np.sum(ew>=0) == ew.size
                HR = ev.T.conj()@HRp1@ev
                bO_r = ev.T.conj()@kron(O_r,eye(MR))@ev
            else:
                HR = HRp1
                bO_r = kron(O_r,eye(MR))
            nL -= 1
            nR += 1
            HR_dict[nR] = HR
            bO_r_dict[nR] = bO_r

    return E0_gs


def finite_dmrg_efficient(N,chi_max,O_os,O_l,O_r,maxiter=50):
    assert N%2==0
    assert N >= 6

    HL_dict, HR_dict, bO_l_dict, bO_r_dict = initialize_dmrg(N=N,m=chi_max,O_os=O_os,O_l=O_l,O_r=O_r)
    
    E0_gs = np.inf
    nL = N//2
    nR = N-nL-2
    for i in range(maxiter):
        while nL <= N-3:
            HL = HL_dict[nL]
            ML = HL.shape[0]
            bO_l = bO_l_dict[nL]
            HR = HR_dict[nR]
            MR = HR.shape[0]
            bO_r = bO_r_dict[nR]
            HLp1 = kron(HL,eye(2)) + kron(bO_l,O_r) + kron(eye(ML),O_os)
            HRp1 = kron(eye(2),HR) + kron(O_l,bO_r) + kron(O_os,eye(MR))
            if 2*ML > chi_max:
                H = kron(HLp1,eye(2*MR)) + kron(eye(2*ML),HRp1) + kron(kron(eye(ML),O_l),kron(O_r,eye(MR)))
                ew, ev = eigsh(H,k=1,which="SA")
                E0 = ew[0]
                if nL == N//2:
                    E0_gs = E0
                    if np.abs(E0 - E0_gs) < 1e-12:
                        print("Converged!")
                        return E0_gs
                gs = ev[:,0]
                gs_m = gs.reshape(2*ML,2*MR)
                u, s, vh = svd(gs_m, k=chi_max, tol=1e-12, which="LM")
                HL = u.T.conj()@HLp1@u
                bO_l = u.T.conj()@kron(eye(ML),O_l)@u
            else:
                H = kron(HLp1,eye(2*MR)) + kron(eye(2*ML),HRp1) + kron(kron(eye(ML),O_l),kron(O_r,eye(MR)))
                ew, ev = eigsh(H,k=1,which="SA")
                E0_gs = ew[0]
                HL = HLp1
                bO_l = kron(eye(ML),O_l)
            nL += 1
            nR -= 1
            HL_dict[nL] = HL
            bO_l_dict[nL] = bO_l

        while nR <= N-3:
            HL = HL_dict[nL]
            ML = HL.shape[0]
            bO_l = bO_l_dict[nL]
            HR = HR_dict[nR]
            MR = HR.shape[0]
            bO_r = bO_r_dict[nR]
            HLp1 = kron(HL,eye(2)) + kron(bO_l,O_r) + kron(eye(ML),O_os)
            HRp1 = kron(eye(2),HR) + kron(O_l,bO_r) + kron(O_os,eye(MR))
            if 2*MR > m:
                H = kron(HLp1,eye(2*MR)) + kron(eye(2*ML),HRp1) + kron(kron(eye(ML),O_l),kron(O_r,eye(MR)))
                ew, ev = eigsh(H,k=1,which="SA")
                gs = ev[:,0]
                gs_m = gs.reshape(2*ML,2*MR).T
                u, s, vh = svd(gs_m, k=m, tol=1e-12, which="LM")
                HR = u.T.conj()@HRp1@u
                bO_r = u.T.conj()@kron(O_r,eye(MR))@u
            else:
                HR = HRp1
                bO_r = kron(O_r,eye(MR))
            nL -= 1
            nR += 1
            HR_dict[nR] = HR
            bO_r_dict[nR] = bO_r

    return E0_gs


def main():
    pz = np.array([[1,0],[0,-1]])
    px = np.array([[0,1],[1,0]])
    initialize_dmrg(N=4, m=4, O_os=pz, O_l=px, O_r=px)

if __name__ == "__main__":
    main()