import numpy as np
from NumericalGroundStateSearch.MPS.MPS import random_MPS, spinup_MPS, spindown_MPS, spinplus_MPS, bond_to_hilbert

PHYS_DIM = 2

p_x = np.zeros((2, 2))
p_x[0, 1] = 1
p_x[1, 0] = 1
p_z = np.zeros((2,2))
p_z[0, 0] = 1
p_z[1, 1] = -1

# For maximal bond dimension the quality
# (overlap between normalized mps and original normalized state)
# should be one.
def test_random_mps():
    for N in range(1,12):
        _, qual = random_MPS(N=N,chi_max=PHYS_DIM**(N//2))
        assert np.abs(qual-1.0) < 1e-14


# Checks that the Hilbertspace representations of the site and
# bond canonical MPSs agree.  
def test_bond_canonical_form():

    for N in range(1,12):
        mps, _ = random_MPS(N=N)
        hilbert_orig = mps.hilbert()
        for i in range(1,N):
            Us, S, Vs = mps.bond_canonical(i)
            hilbert_bc = bond_to_hilbert(Us=Us, S=S, Vs=Vs)
            assert np.allclose(hilbert_orig, hilbert_bc)


def test_spin_up():

    for N in range(2,12):
        mps = spinup_MPS(N=N)
        for i in range(N):
            assert np.abs(mps.n_site_expectation_value(O=p_x, start=i, n=1)) < 1e-14
            assert np.abs(mps.n_site_expectation_value(O=p_z, start=i, n=1) - 1.0) < 1e-14


def test_spin_down():

    for N in range(2,12):
        mps = spindown_MPS(N=N)
        for i in range(N):
            assert np.abs(mps.n_site_expectation_value(O=p_x, start=i, n=1)) < 1e-14
            assert np.abs(mps.n_site_expectation_value(O=p_z, start=i, n=1) + 1.0) < 1e-14


def test_spin_plus():

    for N in range(2,12):
        mps = spinplus_MPS(N=N)
        for i in range(N):
            assert np.abs(mps.n_site_expectation_value(O=p_x, start=i, n=1) - 1.0) < 1e-14
            assert np.abs(mps.n_site_expectation_value(O=p_z, start=i, n=1)) < 1e-14
    

