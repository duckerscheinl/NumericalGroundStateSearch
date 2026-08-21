"""Matrix product states in a redundant, site-centred representation.

Every tensor carries three legs in the order ``(vL, p, vR)``::

        vL --( T )-- vR
                |
                p

An :class:`MPS` keeps three lists that all describe the *same* state:

* ``Us[i]`` -- left isometry on site ``i``        (``i = 0 .. N-2``)
* ``Vs[i]`` -- right isometry on site ``i+1``     (``i = 0 .. N-2``)
* ``Ts[i]`` -- orthogonality centre on site ``i`` (``i = 0 .. N-1``)

so that for every site ``i``::

    |psi> = U_0 ... U_{i-1}  T_i  V_{i+1} ... V_{N-1}.

The ``i``-th bond sits to the *left* of site ``i``: there is a 0-th but no
N-th bond -- for the right edge use :meth:`MPS.left_normalized`.
All indices are 0-based.
"""

import numpy as np

from ..svd_utils import svd_trunc

PHYS_DIM = 2


def is_left_isometry(T):
    """``sum_{vL,p} conj(T[vL,p,a]) T[vL,p,b] == delta_ab``."""
    gram = np.tensordot(T, T.conj(), [[0, 1], [0, 1]])
    return np.allclose(gram, np.identity(T.shape[2]))


def is_right_isometry(T):
    """``sum_{p,vR} T[a,p,vR] conj(T[b,p,vR]) == delta_ab``."""
    gram = np.tensordot(T, T.conj(), [[1, 2], [1, 2]])
    return np.allclose(gram, np.identity(T.shape[0]))

def hilbert_at(index, Us, Ts, Vs):
    """Dense state vector of length ``d**N`` with orthogonality center at index -- used for validation only."""
    N = len(Ts) 
    phi = np.ones((1))
    for i in range(index):
        phi = np.tensordot(phi, Us[i], [i, 0])
    phi = np.tensordot(phi, Ts[index], [index, 0])
    for i in range(index+1, N):
        phi = np.tensordot(phi, Vs[i-1], [i, 0])

    phi = np.reshape(phi, PHYS_DIM**N)
    assert np.allclose(phi.conj() @ phi, 1), "state is not normalised"
    return phi

def validate_sc_forms(Us, Ts, Vs):
    N = len(Ts)
    sc_0 = hilbert_at(0, Us=Us, Ts=Ts, Vs=Vs)
    for i in range(1, N):
        sc_i = hilbert_at(i, Us=Us, Ts=Ts, Vs=Vs)
        assert np.allclose(sc_0, sc_i)


def bond_to_hilbert(Us,S,Vs):
    n_U = len(Us)
    n_V = len(Vs)
    n = n_U + n_V

    hilbert = np.ones((1))
    for i in range(n_U):
        hilbert = np.tensordot(hilbert, Us[i], [i,0])
    hilbert = np.tensordot(hilbert, np.diag(S), [n_U,0])
    for i in range(n_V):
        hilbert = np.tensordot(hilbert, Vs[i], [n_U+i,0])
    hilbert = np.reshape(hilbert, PHYS_DIM**n)

    return hilbert


class MPS:
    """Matrix product state stored in every canonical form at once.

    Parameters
    ----------
    Us, Vs : list of ndarray
        Left/right isometries, ``N-1`` each.
    Ts : list of ndarray
        Orthogonality centres, one per site.
    validate : bool
        Check the isometry conditions on construction.
        Check that all site-canonical forms represent the same state.
        Check normalization.
    """

    def __init__(self, Us, Vs, Ts, validate=True):
        assert len(Us) == len(Ts) - 1, f"expected {len(Ts) - 1} left isometries, got {len(Us)}"
        assert len(Vs) == len(Ts) - 1, f"expected {len(Ts) - 1} right isometries, got {len(Vs)}"
        if validate:
            for i, U in enumerate(Us):
                assert is_left_isometry(U), f"Us[{i}] (site {i}) is not a left isometry"
            for i, V in enumerate(Vs):
                assert is_right_isometry(V), f"Vs[{i}] (site {i + 1}) is not a right isometry"
            validate_sc_forms(Us,Ts,Vs)

        self.N = len(Ts)
        self.Ts = Ts
        self.Us = Us
        self.Vs = Vs

    def copy(self):
        return MPS(
            Us=[U.copy() for U in self.Us],
            Vs=[V.copy() for V in self.Vs],
            Ts=[T.copy() for T in self.Ts],
        )

    # --- tensor access -----------------------------------------------------

    def get_U(self, i):
        """Left isometry on site ``i``."""
        assert i < self.N - 1, f"site {i} carries no left isometry"
        return self.Us[i]

    def get_V(self, i):
        """Right isometry on site ``i`` (stored at ``Vs[i-1]``)."""
        assert i > 0, f"site {i} carries no right isometry"
        return self.Vs[i - 1]

    def get_T(self, i):
        """Orthogonality centre on site ``i``."""
        return self.Ts[i]

    def update_U(self, i, U):
        self.Us[i] = U

    def update_V(self, i, V):
        assert i > 0, f"site {i} carries no right isometry"
        self.Vs[i - 1] = V

    def update_T(self, i, T):
        assert np.abs(np.linalg.norm(T) - 1) < 1e-12 
        self.Ts[i] = T

    # --- local observables -------------------------------------------------

    def n_site_wave_function(self, start, n):
        """Contract ``n`` sites starting at ``start`` into a single tensor.

        Returns legs ``(v_start, p_start, ..., p_{start+n-1}, v_{start+n})``.
        The centre has to sit on ``start`` for this to be orthonormal.
        """
        assert n > 0
        phi = np.copy(self.Ts[start])
        for i in range(1, n):
            # [vs ps..p_{s+i-1} v_{s+i}] [v_{s+i} p_{s+i} v_{s+i+1}]
            #     -> [vs ps..p_{s+i} v_{s+i+1}]
            phi = np.tensordot(phi, self.get_V(start + i), [i + 1, 0])
        return phi

    def n_site_expectation_value(self, O, start, n):
        """``<psi|O|psi>`` for an ``n``-site operator ``O[out_0..out_n-1, in_0..in_n-1]``.

        WARNING: builds the dense ``n``-site wave function -- cost grows as
        ``d**n``, so keep ``n`` small.
        """
        assert n > 0
        phi = self.n_site_wave_function(start, n)

        ket_legs = list(range(1, n + 1))
        in_legs = list(range(n, 2 * n))
        # [vL p_0..p_{n-1} vR] [out_0..out_{n-1} in_0..in_{n-1}]
        #     -> [vL vR out_0..out_{n-1}]
        psi = np.tensordot(phi, O, [ket_legs, in_legs])

        psi_legs = list(range(n + 2))
        bra_legs = [0, n + 1] + list(range(1, n + 1))
        return np.tensordot(psi, phi.conj(), [psi_legs, bra_legs])

    # --- canonical forms ---------------------------------------------------

    def bond_canonical(self, i):
        """Singular values on bond ``i`` plus the isometries left and right of it.
           The bond will be to the left of site i. So i>0.
        """
        assert i > 0, "bond 0 is the left edge remainder."
        assert i < self.N, f"bond {i} does not exist; for the right edge remainder use left_normalized()."

        Us = [np.copy(self.Us[j]) for j in range(i - 1)]
        Vs = [np.copy(self.get_V(j)) for j in range(i + 1, self.N)]

        T = self.Ts[i]
        T = np.reshape(T, (T.shape[0], T.shape[1] * T.shape[2]))
        U, S, V = np.linalg.svd(T, full_matrices=False)

        U = np.tensordot(self.Us[i - 1], U, [2, 0])
        Us.append(U)
        V = np.reshape(V, (V.shape[0], PHYS_DIM, V.shape[1] // PHYS_DIM))
        Vs.insert(0, V)

        assert len(Us) + len(Vs) == self.N
        return Us, S, Vs

    def left_normalized(self):
        """All ``N`` sites as left isometries, plus the remainder on the right edge.
        """
        Us = [np.copy(self.Us[j]) for j in range(self.N - 1)]

        T = self.Ts[self.N - 1]
        T = np.reshape(T, (T.shape[0] * T.shape[1], T.shape[2]))
        U, S, V = np.linalg.svd(T, full_matrices=False)
        U = np.reshape(U, (U.shape[0] // PHYS_DIM, PHYS_DIM, U.shape[1]))
        Us.append(U)

        S = np.tensordot(np.diag(S), V, [1, 0])
        return S, Us

    # --- global quantities -------------------------------------------------

    def overlap(self, psi):
        """``<psi|self>``, contracted through the left-canonical form of both states."""
        assert psi.N == self.N, f"length mismatch: {self.N} vs {psi.N}"

        env = np.ones((1, 1))
        for i in range(self.N - 1):
            # [v v*] [v p w] -> [v* p w]
            env = np.tensordot(env, self.get_U(i), [0, 0])
            # [v* p w] [v* p* w*] -> [w w*]
            env = np.tensordot(env, psi.get_U(i).conj(), [[0, 1], [0, 1]])

        env = np.tensordot(env, self.get_T(self.N - 1), [0, 0])
        env = np.tensordot(env, psi.get_T(self.N - 1).conj(), [[0, 1, 2], [0, 1, 2]])
        return np.real_if_close(env)

    def hilbert(self):
        """Dense state vector of length ``d**N`` -- debugging aid for small ``N``."""
        phi = np.ones((1, 1))
        for i in range(self.N - 1):
            phi = np.tensordot(phi, self.Us[i], [i + 1, 0])
        phi = np.tensordot(phi, self.Ts[self.N - 1], [self.N, 0])

        phi = np.reshape(phi, PHYS_DIM**self.N)
        assert np.allclose(phi.conj() @ phi, 1), "state is not normalised"
        return phi


def product_MPS(N, local_state):
    """Product state ``|chi> ** N`` from a normalised single-site vector."""
    T = np.zeros([1, PHYS_DIM, 1])
    T[0, :, 0] = local_state
    return MPS(
        Us=[T.copy() for _ in range(N - 1)],
        Vs=[T.copy() for _ in range(N - 1)],
        Ts=[T.copy() for _ in range(N)],
    )


def spinup_MPS(N):
    return product_MPS(N, [1.0, 0.0])


def spindown_MPS(N):
    return product_MPS(N, [0.0, 1.0])


def spinplus_MPS(N):
    return product_MPS(N, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])


def random_MPS(N, chi_max=12):
    """Random state, obtained by compressing a dense random vector.

    WARNING: allocates ``2**N`` amplitudes, so this is a small-``N`` helper only.
    """
    assert N > 0, f"N={N}, has to be at least 1."
    assert chi_max > 0, f"chi_max={chi_max}, has to be at least 1."

    phi_random = np.random.rand(PHYS_DIM**N, 1) + 1j * np.random.rand(PHYS_DIM**N, 1)
    phi = np.copy(phi_random)

    # right-to-left sweep: peel one right isometry off the dense vector per site
    Vs = []
    for _ in range(N - 1):
        phi = phi.reshape((phi.shape[0] // PHYS_DIM, phi.shape[1] * PHYS_DIM))
        U, S, Vh = svd_trunc(phi, chi_max=chi_max)
        V = np.reshape(Vh, (Vh.shape[0], PHYS_DIM, Vh.shape[1] // PHYS_DIM))
        assert is_right_isometry(V)
        Vs.append(V)
        phi = U @ np.diag(S)
    Vs.reverse()

    T = np.reshape(phi, (phi.shape[0] // PHYS_DIM, PHYS_DIM, phi.shape[1]))
    Ts = [T / np.linalg.norm(T)]

    # left-to-right sweep: move the centre along and record the left isometries
    Us = []
    for n in range(N - 1):
        T = np.copy(Ts[n])
        T = T.reshape((T.shape[0] * T.shape[1], T.shape[2]))
        U, S, Vh = np.linalg.svd(T, full_matrices=False)
        U = np.reshape(U, (U.shape[0] // PHYS_DIM, PHYS_DIM, U.shape[1]))
        assert is_left_isometry(U)
        Us.append(U)
        Ts.append(np.tensordot(np.diag(S) @ Vh, Vs[n], [1, 0]))

    phi_mps = MPS(Us=Us, Vs=Vs, Ts=Ts)
    phi_hilbert = phi_mps.hilbert()

    phi_test = np.reshape(phi_random, (PHYS_DIM**N))
    phi_test = phi_test/np.linalg.norm(phi_test)

    overlap = np.inner(phi_hilbert.conj(), phi_test)

    return MPS(Us=Us, Vs=Vs, Ts=Ts), overlap
