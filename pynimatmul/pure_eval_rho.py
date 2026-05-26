import numpy as np
from pynimatmul.flags import num_nvar, num_ao_comp


def get_rho_from_dm_with_output(
    ao: np.ndarray,
    dm_list: list[np.ndarray],
    den_type: str,
    out: np.ndarray,
    buf: np.ndarray,
):
    """Evaluate density from density matrices.

    The input density matrices must be symmetric.

    Parameters
    ----------
    ao : np.ndarray
        The AO values with shape [ncomp, nao, ngrid].
    dm_list : list[np.ndarray]
        The list of density matrices, each with shape [nao, nao].
    den_type : str
        The type of density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
    out : np.ndarray
        The output array to store the computed density, with shape [nset, nvar, ngrid].
    buf : np.ndarray
        The buffer array to store intermediate results, with size ngrids * nao.
    """
    assert ao.ndim == 3
    nao = ao.shape[1]

    for dm in dm_list:
        assert dm.ndim == 2
        assert dm.shape == (nao, nao)

    nset = len(dm_list)
    ngrids = ao.shape[2]
    nvar = num_nvar(den_type)
    assert out.shape == (nset, nvar, ngrids)
    assert buf.size >= ngrids * nao
    assert ao.shape[0] >= num_ao_comp(den_type)

    for i, dm in enumerate(dm_list):
        # rho part
        scr = dm @ ao[0]
        out[i, 0] = (scr * ao[0]).sum(axis=0)
        # sigma part
        if den_type in ("SIGMA", "TAU", "LAPL"):
            out[i, 1:4] = 2 * (scr * ao[1:4]).sum(axis=1)
        # lapl part (second derivative of AO)
        if den_type == "LAPL":
            for t in [4, 7, 9]:
                out[i, 5] += 2 * (scr * ao[t]).sum(axis=0)
        # tau part
        if den_type in ("TAU", "LAPL"):
            for t in [1, 2, 3]:
                scr = 0.5 * dm @ ao[t]
                out[i, 4] += (scr * ao[t]).sum(axis=0)
        # lapl part (tau contribution)
        if den_type == "LAPL":
            out[i, 5] += 4 * out[i, 4]
    return out


def get_rho_from_homogeneous_braket_with_output(
    ao: np.ndarray,
    bra_list: list[np.ndarray],
    den_type: str,
    out: np.ndarray,
    buf: np.ndarray,
):
    """Evaluate density from orbital coefficients where bra and ket are the same.

    Parameters
    ----------
    ao : np.ndarray
        The AO values with shape [ncomp, nao, ngrid].
    bra_list : list[np.ndarray]
        Orbital coefficient matrices, each with shape [nao, nocc]; one per set.
    den_type : str
        The type of density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
    out : np.ndarray
        The output array, with shape [nset, nvar, ngrid].
    buf : np.ndarray
        The buffer array, with size at least 2 * ngrid * nocc_max.
    """
    assert ao.ndim == 3
    nao = ao.shape[1]

    for bra in bra_list:
        assert bra.ndim == 2
        assert bra.shape[0] == nao

    nocc_max = max(bra.shape[1] for bra in bra_list) if bra_list else 0
    nset = len(bra_list)
    ngrid = ao.shape[2]
    nvar = num_nvar(den_type)

    assert out.shape == (nset, nvar, ngrid)
    assert buf.size >= 2 * ngrid * nocc_max
    assert ao.shape[0] >= num_ao_comp(den_type)

    for i, bra in enumerate(bra_list):
        # rho part
        scr1 = bra.T @ ao[0]  # [nocc, ngrid]
        out[i, 0] = (scr1 * scr1).sum(axis=0)
        if den_type in ("SIGMA", "TAU", "LAPL"):
            for t in range(1, 4):
                scr2 = bra.T @ ao[t]  # [nocc, ngrid]
                # sigma part
                out[i, t] = 2 * (scr1 * scr2).sum(axis=0)
                # tau part
                if den_type in ("TAU", "LAPL"):
                    out[i, 4] += 0.5 * (scr2 * scr2).sum(axis=0)
        if den_type == "LAPL":
            # lapl part (second derivative of AO)
            for t in [4, 7, 9]:
                scr2 = bra.T @ ao[t]
                out[i, 5] += 2 * (scr1 * scr2).sum(axis=0)
            # lapl part (tau contribution)
            out[i, 5] += 4 * out[i, 4]
    return out


def get_rho_from_one_bra_mult_ket_with_output(
    ao: np.ndarray,
    bra: np.ndarray,
    ket_list: list[np.ndarray],
    den_type: str,
    out: np.ndarray,
    buf: np.ndarray,
):
    """Evaluate density from one shared bra and multiple kets.

    Parameters
    ----------
    ao : np.ndarray
        The AO values with shape [ncomp, nao, ngrid].
    bra : np.ndarray
        Shared orbital coefficient matrix with shape [nao, nocc].
    ket_list : list[np.ndarray]
        Orbital coefficient matrices for each set, each with shape [nao, nocc].
    den_type : str
        The type of density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
    out : np.ndarray
        The output array, with shape [nset, nvar, ngrid].
    buf : np.ndarray
        The buffer array, with size at least 3 * ngrid * nocc.
    """
    assert ao.ndim == 3
    nao = ao.shape[1]

    assert bra.ndim == 2
    assert bra.shape[0] == nao
    nocc = bra.shape[1]

    for ket in ket_list:
        assert ket.ndim == 2
        assert ket.shape[0] == nao
        assert ket.shape[1] == nocc

    nset = len(ket_list)
    ngrid = ao.shape[2]
    nvar = num_nvar(den_type)

    assert out.shape == (nset, nvar, ngrid)
    assert buf.size >= 3 * ngrid * nocc
    assert ao.shape[0] >= num_ao_comp(den_type)

    # Pre-compute scr1 = bra.T @ ao[0] (shared across all sets)
    scr1 = bra.T @ ao[0]  # [nocc, ngrid]

    for i, ket in enumerate(ket_list):
        # rho part
        scr2 = ket.T @ ao[0]  # [nocc, ngrid]
        out[i, 0] = (scr1 * scr2).sum(axis=0)

        # sigma part
        if den_type in ("SIGMA", "TAU", "LAPL"):
            for t in range(1, 4):
                scr3 = ket.T @ ao[t]
                out[i, t] = (scr1 * scr3).sum(axis=0)
                scr3 = bra.T @ ao[t]
                out[i, t] += (scr3 * scr2).sum(axis=0)

        # lapl part (second derivative of AO), must come before tau which overwrites scr2
        if den_type == "LAPL":
            for t in [4, 7, 9]:
                scr3 = ket.T @ ao[t]
                out[i, 5] += (scr1 * scr3).sum(axis=0)
                scr3 = bra.T @ ao[t]
                out[i, 5] += (scr3 * scr2).sum(axis=0)

        # tau part (overwrites scr2, which is no longer needed for sigma/lapl)
        if den_type in ("TAU", "LAPL"):
            for t in range(1, 4):
                scr2 = ket.T @ ao[t]
                scr3 = bra.T @ ao[t]
                out[i, 4] += 0.5 * (scr3 * scr2).sum(axis=0)

        # lapl part (tau contribution)
        if den_type == "LAPL":
            out[i, 5] += 4 * out[i, 4]

    return out


def get_rho_from_mult_bra_mult_ket_with_output(
    ao: np.ndarray,
    bra_list: list[np.ndarray],
    ket_list: list[np.ndarray],
    den_type: str,
    out: np.ndarray,
    buf: np.ndarray,
):
    """Evaluate density from multiple bra-ket pairs.

    Parameters
    ----------
    ao : np.ndarray
        The AO values with shape [ncomp, nao, ngrid].
    bra_list : list[np.ndarray]
        Orbital coefficient matrices for bra, each with shape [nao, nocc_i].
    ket_list : list[np.ndarray]
        Orbital coefficient matrices for ket, each with shape [nao, nocc_i].
    den_type : str
        The type of density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
    out : np.ndarray
        The output array, with shape [nset, nvar, ngrid].
    buf : np.ndarray
        The buffer array, with size at least 3 * ngrid * nocc_max.
    """
    assert ao.ndim == 3
    nao = ao.shape[1]

    assert len(bra_list) == len(ket_list)
    nocc_max = max(bra.shape[1] for bra in bra_list) if bra_list else 0

    for bra, ket in zip(bra_list, ket_list):
        assert bra.ndim == 2
        assert ket.ndim == 2
        assert bra.shape[0] == nao
        assert ket.shape[0] == nao
        assert bra.shape[1] == ket.shape[1]

    nset = len(bra_list)
    ngrid = ao.shape[2]
    nvar = num_nvar(den_type)

    assert out.shape == (nset, nvar, ngrid)
    assert buf.size >= 3 * ngrid * nocc_max
    assert ao.shape[0] >= num_ao_comp(den_type)

    for i, (bra, ket) in enumerate(zip(bra_list, ket_list)):
        # rho part
        scr1 = bra.T @ ao[0]  # [nocc, ngrid]
        scr2 = ket.T @ ao[0]  # [nocc, ngrid]
        out[i, 0] = (scr1 * scr2).sum(axis=0)

        # sigma part
        if den_type in ("SIGMA", "TAU", "LAPL"):
            for t in range(1, 4):
                scr3 = ket.T @ ao[t]
                out[i, t] = (scr1 * scr3).sum(axis=0)
                scr3 = bra.T @ ao[t]
                out[i, t] += (scr3 * scr2).sum(axis=0)

        # lapl part (second derivative of AO), must come before tau which overwrites scr1/scr2
        if den_type == "LAPL":
            for t in [4, 7, 9]:
                scr3 = ket.T @ ao[t]
                out[i, 5] += (scr1 * scr3).sum(axis=0)
                scr3 = bra.T @ ao[t]
                out[i, 5] += (scr3 * scr2).sum(axis=0)

        # tau part (overwrites scr1/scr2, which are no longer needed for sigma/lapl)
        if den_type in ("TAU", "LAPL"):
            for t in range(1, 4):
                scr1 = bra.T @ ao[t]
                scr2 = ket.T @ ao[t]
                out[i, 4] += 0.5 * (scr1 * scr2).sum(axis=0)

        # lapl part (tau contribution)
        if den_type == "LAPL":
            out[i, 5] += 4 * out[i, 4]

    return out
