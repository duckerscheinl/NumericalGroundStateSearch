"""
Tests for the contraction and effective-operator helpers in mps_utils.
The reference full hilbert space costs
``D**n``, which is why every case stays small, but it shares no index arithmetic
with the code under test.
!Entirely written by Claude!
"""

import numpy as np
import pytest

# adjust to the actual sub-package, e.g. NumericalGroundStateSearch.dmrg.mps_utils
from NumericalGroundStateSearch.MPS.mps_utils import (
    eff_1s_op_sc,
    eff_1s_op_sc_v2,
    eff_1s_opm_sc,
    left_contraction,
    right_contraction,
)

D = 2  # physical dimension

# (len(Us), len(Vs)) splits.  Both zeros is a one-site chain, and the two
# one-sided cases exercise the shift/branching in eff_1s_op_sc.
SPLITS = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (1, 2), (0, 3), (3, 0), (2, 2)]


@pytest.fixture
def rng():
    """Fresh, seeded generator per test, so a failure is reproducible."""
    return np.random.default_rng(20240614)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def random_tensor(chi_l, chi_r, rng):
    """Generic complex site tensor [vL p vR], no isometry property."""
    shape = (chi_l, D, chi_r)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def left_isometry(chi_l, chi_r, rng):
    """[vL p vR] with ``sum_{vL,p} conj(T) T = identity``, exact to machine precision."""
    assert chi_r <= chi_l * D
    Q, _ = np.linalg.qr(random_tensor(chi_l, chi_r, rng).reshape(chi_l * D, chi_r))
    return Q.reshape(chi_l, D, chi_r)


def right_isometry(chi_l, chi_r, rng):
    """[vL p vR] with ``sum_{p,vR} T conj(T) = identity``, exact to machine precision."""
    assert chi_l <= D * chi_r
    Q, _ = np.linalg.qr(random_tensor(chi_r, chi_l, rng).reshape(D * chi_r, chi_l))
    return Q.T.reshape(chi_l, D, chi_r)


def dense_state(tensors):
    """Chain of [vL p vR] tensors with trivial edge bonds -> state vector."""
    psi = np.ones((1, 1))
    for k, T in enumerate(tensors):
        psi = np.tensordot(psi, T, [k + 1, 0])
    return psi.reshape(-1)


def dense_operator(op, n):
    """Legs [0 .. n-1, 0* .. n-1*] -> matrix, rows = unstarred (bra) indices."""
    return op.reshape(D**n, D**n)


def random_operator(n, rng, hermitian=False):
    M = rng.normal(size=(D**n, D**n)) + 1j * rng.normal(size=(D**n, D**n))
    if hermitian:
        M = M + M.conj().T
    return M.reshape((D,) * (2 * n))


def identity_operator(n):
    return np.eye(D**n).reshape((D,) * (2 * n))


def left_isometries(n, rng, chi=3):
    """``n`` left isometries with bond dimensions 1, ..., chi."""
    dims = [1] + [min(chi, D**k) for k in range(1, n + 1)]
    return [left_isometry(dims[k], dims[k + 1], rng) for k in range(n)]


def right_isometries(n, rng, chi=3):
    """``n`` right isometries with bond dimensions chi, ..., 1."""
    dims = [min(chi, D**k) for k in range(n, 0, -1)] + [1]
    return [right_isometry(dims[k], dims[k + 1], rng) for k in range(n)]


def isometric_chain(nU, nV, rng, chi=3):
    """Left isometries, a normalised centre tensor, right isometries.

    The centre is the only non-isometric tensor, which is exactly the situation
    the eff_1s_* functions assume.
    """
    Us = left_isometries(nU, rng, chi)
    Vs = right_isometries(nV, rng, chi)
    chi_l = Us[-1].shape[2] if Us else 1
    chi_r = Vs[0].shape[0] if Vs else 1
    T = random_tensor(chi_l, chi_r, rng)
    return Us, T / np.linalg.norm(T), Vs


# ---------------------------------------------------------------------------
# left_contraction / right_contraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chis", [[1, 3, 1], [1, 2, 3, 1], [1, 2, 4, 2, 1], [1, 2, 4, 4, 2, 1]])
def test_contraction_equals_dense_expectation(chis, rng):
    """Both contractions reproduce <psi|O|psi> over a whole chain.

    Uses generic, *non-isometric* tensors on purpose.  The docstrings promise the
    contraction is exact for arbitrary tensors and that the isometry condition is
    about interpretation, not correctness -- this is where that promise is pinned
    down.  With trivial edge bonds both functions close completely, so the open
    bond is 1x1 and the entry is the expectation value.
    """
    n = len(chis) - 1
    tensors = [random_tensor(chis[k], chis[k + 1], rng) for k in range(n)]
    op = random_operator(n, rng)

    psi = dense_state(tensors)
    expected = psi.conj() @ dense_operator(op, n) @ psi

    for contraction in (left_contraction, right_contraction):
        result = contraction(tensors, op)
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], expected)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_contraction_of_identity_is_the_block_metric(n, rng):
    """With the identity operator the result is the overlap matrix of the block.

    It equals the identity exactly when the tensors have the handedness the
    docstring demands.  The two negative assertions are what keep this honest: a
    function that ignored its input and returned an identity would pass the first
    half, so the test also insists the *wrong* handedness does not.
    """
    Us = left_isometries(n, rng)
    Vs = right_isometries(n, rng)
    ident = identity_operator(n)

    assert np.allclose(left_contraction(Us, ident), np.eye(Us[-1].shape[2]))
    assert np.allclose(right_contraction(Vs, ident), np.eye(Vs[0].shape[0]))

    assert not np.allclose(left_contraction(Vs, ident), np.eye(Vs[-1].shape[2]))
    assert not np.allclose(right_contraction(Us, ident), np.eye(Us[0].shape[0]))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_contraction_open_bond_sits_next_to_the_centre(n, rng):
    """left_contraction leaves [vn vn*], right_contraction leaves [v0 v0*].

    A cheap shape check, but it is the leg the caller contracts the centre tensor
    against, so a silent transposition of the output would be expensive later.
    """
    Us = left_isometries(n, rng)
    Vs = right_isometries(n, rng)
    op = random_operator(n, rng)

    chi_right_of_left_isometries = Us[-1].shape[2]
    chi_left_of_right_isometries = Vs[0].shape[0]
    assert left_contraction(Us, op).shape == (chi_right_of_left_isometries,) * 2
    assert right_contraction(Vs, op).shape == (chi_left_of_right_isometries,) * 2


@pytest.mark.parametrize("n_trailing", [1, 2])
def test_identity_tail_may_be_dropped_from_the_block(n_trailing, rng):
    """Sites past the operator's support fall out of the contraction.

    This is the documented assumption that the tensors to the right of the
    expression are right isometries.  Because they close to the identity, a block
    padded with identity operators on its last sites has to give the same result
    as the shorter block carrying only the operator's support -- which is what
    lets one contraction serve many one-body operators at different sites.
    """
    n = 4
    Vs = right_isometries(n, rng)
    support = n - n_trailing

    reduced = random_operator(support, rng)
    padded = np.kron(dense_operator(reduced, support), np.eye(D**n_trailing))
    padded = padded.reshape((D,) * (2 * n))

    assert np.allclose(right_contraction(Vs, padded), right_contraction(Vs[:support], reduced))


# ---------------------------------------------------------------------------
# effective one-site operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nU,nV", SPLITS)
def test_eff_op_reproduces_the_expectation_value(nU, nV, rng):
    """The defining property: <T|H_eff|T> == <psi|O|psi>.

    This covers the whole pipeline at once -- the leg bookkeeping in
    eff_1s_op_sc, the transpose in eff_1s_opm_sc, and the flattening convention
    that the matrix acts on ``T.reshape(-1)``.

    The operator is deliberately *not* Hermitian.  For Hermitian H the value of
    <T|H|T> is unchanged if H is accidentally transposed, so a Hermitian operator
    would leave the row/column convention completely untested.
    """
    Us, T, Vs = isometric_chain(nU, nV, rng)
    n = nU + 1 + nV
    op = random_operator(n, rng)

    psi = dense_state(Us + [T] + Vs)
    expected = psi.conj() @ dense_operator(op, n) @ psi

    H = eff_1s_opm_sc(nU, op, Us, Vs)
    t = T.reshape(-1)
    assert H.shape == (t.size, t.size)
    assert np.isclose(t.conj() @ H @ t, expected)


@pytest.mark.parametrize("nU,nV", SPLITS)
def test_eff_op_versions_agree(nU, nV, rng):
    """Both contraction orders must give the identical tensor, leg for leg.

    Keeps the cheaper implementation honest if one of them is later optimised,
    and is the only test that would notice a divergence that happens to leave
    every expectation value intact.
    """
    Us, _, Vs = isometric_chain(nU, nV, rng)
    op = random_operator(nU + 1 + nV, rng)
    assert np.allclose(eff_1s_op_sc(nU, op, Us, Vs), eff_1s_op_sc_v2(nU, op, Us, Vs))


@pytest.mark.parametrize("nU,nV", SPLITS)
def test_eff_op_of_identity_is_identity(nU, nV, rng):
    """The environment embeds the centre tensor without changing its norm.

    So the effective operator of the identity is the identity matrix.  This is
    the effective-operator counterpart of the block-metric test, and it is the
    check that fails loudest if an isometry is ever absorbed with the wrong
    conjugation: the norm would stop being preserved.
    """
    Us, T, Vs = isometric_chain(nU, nV, rng)
    H = eff_1s_opm_sc(nU, identity_operator(nU + 1 + nV), Us, Vs)
    assert np.allclose(H, np.eye(T.size))


@pytest.mark.parametrize("nU,nV", SPLITS)
def test_eff_op_is_hermitian_and_variational(nU, nV, rng):
    """H_eff = P^dag O P with P an isometry, so two things follow.

    A Hermitian operator gives a Hermitian H_eff, which is what makes it legal
    for a ground-state solver to call eigh rather than eig.  And the effective
    spectrum cannot dip below the true one -- the variational property the whole
    sweep relies on.  A sign or conjugation error in the environment tends to
    break the bound even when Hermiticity survives.
    """
    Us, _, Vs = isometric_chain(nU, nV, rng)
    n = nU + 1 + nV
    op = random_operator(n, rng, hermitian=True)

    H = eff_1s_opm_sc(nU, op, Us, Vs)
    assert np.allclose(H, H.conj().T)
    assert np.linalg.eigvalsh(H)[0] >= np.linalg.eigvalsh(dense_operator(op, n))[0] - 1e-10


def test_eff_op_rejects_an_inconsistent_split(rng):
    """i, len(Us) and len(Vs) have to describe one and the same chain.

    Cheap, but these asserts are the only thing standing between a mis-specified
    centre and a contraction that happens to have matching shapes and returns
    quiet nonsense.
    """
    Us, _, Vs = isometric_chain(2, 2, rng)
    op = random_operator(5, rng)

    with pytest.raises(AssertionError):
        eff_1s_op_sc(1, op, Us, Vs)  # i disagrees with len(Us)
    with pytest.raises(AssertionError):
        eff_1s_op_sc(2, op, Us, Vs[:-1])  # Vs too short for the operator width