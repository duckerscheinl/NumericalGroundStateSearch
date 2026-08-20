import numpy as np


def right_contraction(Vs: list, operator: np.ndarray):
    """
    Computes the contraction of a partial state with an operator. \\
    It is assumed that all tensors to the right of the expression are right
    isometries. \\ 
    The Vs do not technically need to be right isometries -- the contraction is
    exact for arbitrary tensors -- but they have to be for the result to be part of
    the right environment. The orthogonality center sits to the left of this
    expression.\\
    Assumed structure of site-i-tensor: [vi pi vi+1]

    Parameters
    ----------
    Vs : list[np.ndarray]
        List of n site-tensors of the ket-state. 
    operator : np.ndarray
        n-site operator. Leg order [i i+1 i* i+1*] (Two-site example).

    Returns
    -------
    np.ndarray
        The contraction of the tensors -> matrix. [vR+1 vR+1*] (for R: site to the left of the expression)
    """

    assert operator.ndim // 2 == len(Vs)
    n = len(Vs)

    contr = np.tensordot(Vs[n-1], operator, [1,2*n-1]) # [vn-1 pn-1 vn] [0 ... n-1 0* ... n-1*] -> [vn-1 vn 0 ... n-1 0* ... n-2*]
    contr = np.tensordot(contr, Vs[n-1].conj(), [[1,n+1],[2,1]]) # [vn-1 vn 0 ... n-1 0* ... n-2*] [vn-1* pn-1* vn*] -> [vn-1 0 ... n-2 0* ... n-2* vn-1*]

    for i in range(1,n):
        contr = np.tensordot(Vs[n-1-i], contr, [[1,2],[2*(n-i),0]]) # [vn-i-1 pn-i-1 vn-i] [vn-i 0 ... n-i-1 0* ... n-i-1* vn-i*] -> [vn-i-1 0 ... n-i-1 0* ... n-i-2* vn-i*]
        contr = np.tensordot(contr, Vs[n-1-i].conj(), [[n-i,2*(n-i)],[1,2]]) # [vn-i-1 0 ... n-i-1 0* ... n-i-2* vn-i*] [vn-i-1* pn-i-1* vn-i*] -> [vn-i-1 0 ... n-i-2 0* ... n-i-2* vn-i-1*]

    return contr


def left_contraction(Us: list, operator: np.ndarray):
    """
    Computes the contraction of a partial state with an operator. \\
    It is assumed that all tensors to the left of the expression are left
    isometries. \\
    The Us do not technically need to be left isometries -- the contraction is
    exact for arbitrary tensors -- but they have to be for the result to be part of
    the left environment. The orthogonality center sits to the right of this
    expression.\\
    Assumed structure of site-i-tensor: [vi pi vi+1]

    Parameters
    ----------
    Us : list[np.ndarray]
        List of n site-tensors of the ket-state. 
    operator : np.ndarray
        n-site operator. Leg order [i i+1 i* i+1*] (Two-site example).

    Returns
    -------
    np.ndarray
        The contraction of the tensors -> matrix. [vL vL*] (for L: site to the right of the expression.)
    """

    assert operator.ndim // 2 == len(Us)
    n = len(Us)

    contr = np.tensordot(Us[0], operator, [1,n]) # [v0 p0 v1] [0 ... n-1 0* ... n-1*] -> [v0 v1 0 ... n-1 1* ... n-1*]
    contr = np.tensordot(contr, Us[0].conj(), [[0,2],[0,1]]) # [v0 v1 0 ... n-1 1* ... n-1*] [v0* p0* v1*] -> [v1 1 ... n-1 1* ... n-1* v1*]

    for i in range(1,n):
        contr = np.tensordot(Us[i], contr, [[0,1],[0,n-i+1]]) # [vi pi vi+1] [vi i ... n-1 i* ... n-1* vi*] -> [vi+1 i ... n-1 i+1* ... n-1* vi*]
        contr = np.tensordot(contr, Us[i].conj(), [[2*(n-i),1],[0,1]]) # [vi+1 i ... n-1 i+1* ... n-1* vi*] [vi* pi* vi+1*] -> [vi+1 i+1 ... n-1 i+1* ... n-1* vi+1*]

    return contr


def eff_1s_op_sc(i: int, op: np.ndarray, Us: list, Vs: list):
    """
    Computes the effective one-site operator on site ``i``. \\
    The operator is assumed to be effective regarding the single non-isometry
    site, i.e. every site left of ``i`` enters as a left-isometry and every site
    right of ``i`` as a right-isometry. The isometries are absorbed one site at a
    time, from the outside inwards.\\
    Assumed structure of site-i-tensor: [vi pi vi+1]

    Parameters
    ----------
    i : int
        Site the effective operator acts on. Has to equal ``len(Us)``.
    op : np.ndarray
        n-site operator. Leg order [i i+1 i* i+1*] (Two-site example).
    Us : list[np.ndarray]
        Left-isometries of the sites 0 ... i-1, in ascending site order.
    Vs : list[np.ndarray]
        Right-isometries of the sites i+1 ... n-1, in ascending site order.

    Returns
    -------
    np.ndarray
        Effective operator of form: [vi+1 vi i i* vi* vi+1*]. \\
        An empty ``Us`` drops the [vi vi*] pair, an empty ``Vs`` drops the
        [vi+1 vi+1*] pair, so the result carries 6, 4 or 2 legs.
    """

    assert len(Us) == i    
    assert len(Vs) == (op.ndim // 2) - i - 1

    n = op.ndim // 2
    nU = len(Us)
    nV = len(Vs)

    # An empty left block leaves no [vnU vnU*] pair in front of the operator legs,
    # so every index into the right block moves down by one.
    shift = (nU == 0)
    eff_op = op
    if nU > 0:
        eff_op = np.tensordot(Us[0], eff_op, [1,n]) # [v0 p0 v1] [0 ... n-1 0* ... n-1*] -> [v0 v1 0 ... n-1 1* ... n-1*]
        eff_op = np.tensordot(eff_op, Us[0].conj(), [[0,2],[0,1]]) # [v0 v1 0 ... n-1 1* ... n-1*] [v0* p0* v1*] -> [v1 1 ... n-1 1* ... n-1* v1*]
    for i in range(1,nU):
        eff_op = np.tensordot(Us[i], eff_op, [[0,1],[0,n-i+1]]) # [vi pi vi+1] [vi i ... n-1 i* ... n-1* vi*] -> [vi+1 i ... n-1 i+1* ... n-1* vi*]
        eff_op = np.tensordot(eff_op, Us[i].conj(), [[1,2*(n-i)],[1,0]]) # [vi+1 i ... n-1 i+1* ... n-1* vi*] [vi* pi* vi+1*] -> [vi+1 i+1 ... n-1 i+1* ... n-1* vi+1*]

    if nV > 0:
        eff_op = np.tensordot(Vs[nV-1], eff_op, [1,2*(n-nU)-shift]) # [vn-1 pn-1 vn] [vnU nU ... n-1 nU* ... n-1* vnU*] -> [vn-1 vn vnU nU ... n-1 nU* ... n-2* vnU*]
        eff_op = np.tensordot(eff_op, Vs[nV-1].conj(), [[1,n-nU+2-shift],[2,1]]) # [vn-1 vn vnU nU ... n-1 nU* ... n-2* vnU*] [vn-1* pn-1* vn*] -> [vn-1 vnU nU ... n-2 nU* ... n-2* vnU* vn-1*]
    for i in range(1,nV):
        eff_op = np.tensordot(Vs[nV-1-i], eff_op, [[1,2],[2*(n-nU-i)+1-shift,0]]) # [vn-1-i pn-1-i vn-i] [vn-i vnU nU ... n-1-i nU* ... n-1-i* vnU* vn-i*] -> [vn-1-i vnU nU ... n-1-i nU* ... n-i-2* vnU* vn-i*]
        eff_op = np.tensordot(eff_op, Vs[nV-i-1].conj(), [[n-nU-i+1-shift,2*(n-nU-i+1-shift)],[1,2]]) # [vn-1-i vnU nU ... n-1-i nU* ... n-i-2* vnU* vn-i*] [vn-1-i* pn-1-i* vn-i*] -> [vn-1-i vnU nU ... n-2-i nU* ... n-i-2* vnU* vn-i-1*]
    
    return eff_op


def eff_1s_opm_sc(i: int, op: np.ndarray, Us: list, Vs: list):
    """
    Matricises the effective one-site operator of :func:`eff_1s_op_sc`. \\
    Rows collect the bra-legs, columns the ket-legs, each in the leg order of a
    site tensor, so that the matrix acts on the flattened centre tensor
    ``T.reshape(-1)`` with ``T`` of form [vi pi vi+1].\\
    Assumed structure of site-i-tensor: [vi pi vi+1]

    Parameters
    ----------
    i : int
        Site the effective operator acts on. Has to equal ``len(Us)``.
    op : np.ndarray
        n-site operator. Leg order [i i+1 i* i+1*] (Two-site example).
    Us : list[np.ndarray]
        Left-isometries of the sites 0 ... i-1, in ascending site order.
    Vs : list[np.ndarray]
        Right-isometries of the sites i+1 ... n-1, in ascending site order.

    Returns
    -------
    np.ndarray
        The effective operator as a matrix, rows [vi* i vi+1*] against
        columns [vi i* vi+1]. \\
        A missing left or right block drops the corresponding bond leg from both,
        cf. :func:`eff_1s_op_sc`.
    """
    eff_op = eff_1s_op_sc(i, op, Us, Vs) # vnU+1 vnU nU nU* vnU* vnU+1*
    nU = len(Us)
    nV = len(Vs)
    if nU > 0 and nV > 0:
        eff_op = np.transpose(eff_op, [4,2,5,1,3,0]) # -> [vnU* nU vnU+1* vnU nU* vnU+1]
        return np.reshape(eff_op, (eff_op.shape[0]*eff_op.shape[1]*eff_op.shape[2],eff_op.shape[3]*eff_op.shape[4]*eff_op.shape[5]))
    if nU > 0 and nV == 0:
        eff_op = np.transpose(eff_op, [3,1,0,2]) # [vnU nU nU* vnU*] -> [vnU* nU vnU nU*]
        return np.reshape(eff_op, (eff_op.shape[0]*eff_op.shape[1],eff_op.shape[2]*eff_op.shape[3]))
    if nU == 0 and nV > 0:
        eff_op = np.transpose(eff_op, [1,3,2,0]) # [vnU+1 nU nU* vnU+1*] -> [nU vnU+1* nU* vnU+1]
        return np.reshape(eff_op, (eff_op.shape[0]*eff_op.shape[1],eff_op.shape[2]*eff_op.shape[3]))

    # Case nU == 0, nV == 0
    return eff_op
        

    

def eff_1s_op_sc_v2(i: int, op: np.ndarray, Us: list, Vs: list):
    """
    Computes the effective one-site operator on site ``i``, as
    :func:`eff_1s_op_sc` does, but with a different contraction order. \\
    All left-isometries are first contracted into one tensor alpha and all
    right-isometries into one tensor beta, each of which is then absorbed into the
    operator in a single step.\\
    Assumed structure of site-i-tensor: [vi pi vi+1]

    Parameters
    ----------
    i : int
        Site the effective operator acts on. Has to equal ``len(Us)``.
    op : np.ndarray
        n-site operator. Leg order [i i+1 i* i+1*] (Two-site example).
    Us : list[np.ndarray]
        Left-isometries of the sites 0 ... i-1, in ascending site order.
    Vs : list[np.ndarray]
        Right-isometries of the sites i+1 ... n-1, in ascending site order.

    Returns
    -------
    np.ndarray
        Effective operator of form: [vi+1 vi i i* vi* vi+1*], identical to the
        result of :func:`eff_1s_op_sc`. \\
        An empty ``Us`` drops the [vi vi*] pair, an empty ``Vs`` drops the
        [vi+1 vi+1*] pair, so the result carries 6, 4 or 2 legs.
    """

    assert len(Us) == i    
    assert len(Vs) == (op.ndim // 2) - i - 1

    n = op.ndim // 2
    nU = len(Us)
    nV = len(Vs)

    # An empty left block leaves no [vnU vnU*] pair in front of the operator legs,
    # so every index into the right block moves down by one.
    shift = (nU == 0)
    eff_op = np.copy(op)
    if nU > 0:
        alpha = np.copy(Us[0])
        for i in range(1,nU):
            alpha = np.tensordot(alpha, Us[i], [i+1,0]) # [v0 p0 ... pi-1 vi] [vi pi vi+1] -> [v0 p0 ... pi vi+1]
        pid = [i+1 for i in range(nU)]
        oid = [i+n for i in range(nU)]
        eff_op = np.tensordot(alpha, eff_op, [pid,oid]) # [v0 p0 ... pnU-1 vnU] [0 ... nU-1 nU ... n-1 0* ... nU-1* nU* ... n-1*] -> [v0 vnU 0 ... n-1 nU* ... n-1*]
        oid = [i+2 for i in range(nU)]
        eff_op = np.tensordot(eff_op, alpha.conj(), [[0,*(oid)],[0,*(pid)]]) # [v0 vnU 0 ... n-1 nU* ... n-1*] [v0* p0* ... pnU-1* vnU*] -> [vnU nU ... n-1 nU* ... n-1* vnU*]
    if nV > 0:
        beta = np.copy(Vs[nV-1])
        for i in range(1,nV):
            beta = np.tensordot(Vs[nV-1-i], beta, [2,0]) # [vn-1-i pn-1-i vn-i] [vn-i pn-i ... pn-1 vn] -> [vn-1-i pn-1-i ... pn-1 vn]
        pid = [i+1 for i in range(nV)]
        oid = [n-nU+2+i-shift for i in range(nV)]
        eff_op = np.tensordot(beta, eff_op, [pid,oid]) # [vn-nV pn-nV ... pn-1 vn] [vnU nU ... n-1 nU* ... n-1* vnU*] -> [vn-nV vn vnU nU ... n-1 nU* vnU*]
        oid = [i+4-shift for i in range(nV)]
        eff_op = np.tensordot(eff_op, beta.conj(), [[1,*(oid)],[nV+1,*(pid)]]) # [vn-nV vn vnU nU ... n-1 nU* vnU*] [vn-nV* pn-nV* ... pn-1* vn*] -> [vn-nV vnU nU nU* vnU* vn-nV*]

    return eff_op