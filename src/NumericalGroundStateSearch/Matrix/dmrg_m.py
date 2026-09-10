import numpy as np
from ..numpy_extensions import svd_trunc, gs_h
from numpy import kron, eye


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