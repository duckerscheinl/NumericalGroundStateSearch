import numpy as np
import pytest
from test_mps_utils import random_operator, dense_operator
from NumericalGroundStateSearch.MPS.dmrg import dmrg, lowest_energy_subspace


def hamiltonian(n, op_1s, op_2s):

    m_2s = dense_operator(op_2s, 2)
    ham = op_1s
    for i in range(1,n):
        ham = np.kron(ham, np.identity(2))
        ham += np.kron(np.identity(2**i), op_1s)
        ham += np.kron(np.identity(2**(i-1)), m_2s)

    return ham


@pytest.fixture
def rng():
    """Fresh, seeded generator per test, so a failure is reproducible."""
    return np.random.default_rng(20240614)


@pytest.mark.parametrize("n", [2,3,4,5,6,7,8,9])
def test_ss_dmrg(n, rng):
    chi_max = 2**(n//2)
    op_2s = random_operator(n=2, rng=rng, hermitian=True)
    op_1s = random_operator(n=1, rng=rng, hermitian=True)
    energ_dmrg, psi_dmrg = dmrg(N=n, m=chi_max, O1=op_1s, O2=op_2s, method=1)
    ham = hamiltonian(n=n, op_1s=op_1s, op_2s=op_2s)
    energ_exact, psis_exact, dim = lowest_energy_subspace(H=ham)

    # 1e-11: one order above the 1e-12 DMRG convergence cutoff
    if dim > 0:
        # Even for dim = 1 this treatment is preferrable 
        # since it circumvents dealing with the phase factor.
        c, err, _, _ = np.linalg.lstsq(psis_exact, psi_dmrg.hilbert())
        assert np.abs(err[0]) < 1e-11
    else:
        assert False, "Exact solver failed! (dim < 1)"

    assert np.abs(energ_dmrg-energ_exact) < 1e-11