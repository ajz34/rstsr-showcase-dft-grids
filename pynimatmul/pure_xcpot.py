import numpy as np
from pynimatmul.flags import num_ao_comp, num_nvar


def contract_ao_wv(den_type: str, wv: np.ndarray, ao: np.ndarray) -> np.ndarray:
    """Contract AO with wv for RHO/SIGMA/TAU.

    Parameters
    ----------
    den_type : str
        The type of density to compute. Can be "RHO", "SIGMA", or "TAU".
    wv : np.ndarray
        The weight vector, shape [nvar, ngrids].
    ao : np.ndarray
        The AO values, shape [ncomp, nao, ngrids].

    Returns
    -------
    contracted : np.ndarray
        The contracted AO, shape [nao, nao].
    """
    nvar, ngrids = wv.shape
    assert num_nvar(den_type) == nvar
    assert ao.ndim == 3
    nao = ao.shape[1]
    assert ao.shape[2] == ngrids
    assert ao.shape[0] >= num_ao_comp(den_type)

    contracted = np.zeros((nao, nao))
    # RHO contribution
    contracted += 0.5 * ao[0] * wv[0] @ ao[0].T
    # SIGMA contribution
    if den_type in ("SIGMA", "TAU"):
        contracted += ao[1] * wv[1] @ ao[0].T
        contracted += ao[2] * wv[2] @ ao[0].T
        contracted += ao[3] * wv[3] @ ao[0].T
    # TAU contribution
    if den_type in ("TAU",):
        contracted += 0.25 * ao[1] * wv[4] @ ao[1].T
        contracted += 0.25 * ao[2] * wv[4] @ ao[2].T
        contracted += 0.25 * ao[3] * wv[4] @ ao[3].T
    contracted += contracted.swapaxes(-1, -2)
    return contracted


def rks_vxc_pot_with_eff(
    den_type: str,
    vxc_eff: np.ndarray,
    ao: np.ndarray,
    weights: np.ndarray,
):
    """Evaluate XC potential (1st order) with vxc_eff.

    Parameters
    ----------
    den_type : str
        The type of electron density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
    vxc_eff : np.ndarray
        The effective XC potential, shape [nvar, ngrids].
    ao : np.ndarray
        The basis functions, shape [ncomp, nao, ngrids].
    weights : np.ndarray
        The integration weights, shape [ngrids].
    """
    nvar, ngrids = vxc_eff.shape
    assert weights.shape == (ngrids,)
    assert num_nvar(den_type) == nvar
    assert ao.ndim == 3
    assert ao.shape[2] == ngrids
    assert ao.shape[0] >= num_ao_comp(den_type)

    if den_type == "LAPL":
        raise NotImplementedError

    vxc = contract_ao_wv(den_type, weights * vxc_eff, ao)
    return vxc


def rks_fxc_pot_with_eff(
    den_type: str,
    fxc_eff: np.ndarray,
    rho1: np.ndarray,
    ao: np.ndarray,
    weights: np.ndarray,
):
    """Evaluate XC potential (2nd order) with fxc_eff.

    Parameters
    ----------
    den_type : str
        The type of electron density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
    fxc_eff : np.ndarray
        The effective XC kernel, shape [nvar, nvar, ngrids].
    rho1 : np.ndarray
        The first-order density response, shape [nset, nvar, ngrids].
    ao : np.ndarray
        The basis functions, shape [ncomp, nao, ngrids].
    weights : np.ndarray
        The integration weights, shape [ngrids].

    Returns
    -------
    fxc : np.ndarray
        The second-order XC potential, shape [nset, nao, nao].
    """
    nset, nvar, ngrids = rho1.shape
    assert fxc_eff.shape == (nvar, nvar, ngrids)
    assert weights.shape == (ngrids,)
    assert ao.ndim == 3
    nao = ao.shape[1]
    assert ao.shape[2] == ngrids
    assert ao.shape[0] >= num_ao_comp(den_type)

    fxc = np.zeros((nset, nao, nao))
    for i in range(nset):
        fxc_contract = weights * (fxc_eff * rho1[i, None, :, :]).sum(axis=-2)
        fxc[i] = contract_ao_wv(den_type, fxc_contract, ao)
    return fxc


def rks_kxc_pot_with_eff(
    den_type: str,
    kxc_eff: np.ndarray,
    rho1: np.ndarray,
    rho2: np.ndarray,
    ao: np.ndarray,
    weights: np.ndarray,
):
    """Evaluate XC potential (3rd order) with kxc_eff.

    Parameters
    ----------
    den_type : str
        The type of electron density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
    kxc_eff : np.ndarray
        The effective XC kernel, shape [nvar, nvar, nvar, ngrids].
    rho1 : np.ndarray
        The first-order density response, shape [nset-1, nvar, ngrids].
    rho2 : np.ndarray
        The second-order density response, shape [nset-2, nvar, ngrids].
    ao : np.ndarray
        The basis functions, shape [ncomp, nao, ngrids].
    weights : np.ndarray
        The integration weights, shape [ngrids].

    Returns
    -------
    kxc : np.ndarray
        The third-order XC potential, shape [nset-2, nset-1, nao, nao].
        Note the first dimension is the `nset-2` from rho2 in row-major order.
    """
    nset1, nvar, ngrids = rho1.shape
    nset2, _, _ = rho2.shape
    assert kxc_eff.shape == (nvar, nvar, nvar, ngrids)
    assert rho2.shape == (nset2, nvar, ngrids)
    assert weights.shape == (ngrids,)
    assert ao.ndim == 3
    nao = ao.shape[1]
    assert ao.shape[2] == ngrids
    assert ao.shape[0] >= num_ao_comp(den_type)

    kxc = np.zeros((nset2, nset1, nao, nao))
    for i2 in range(nset2):
        for i1 in range(nset1):
            # kxc_eff: [nvar-0, nvar-2, nvar-1, ngrids]
            kxc_contract = weights * (
                kxc_eff * rho1[i1, None, None, :, :] * rho2[i2, None, :, None, :]
            ).sum(axis=(-3, -2))
            kxc[i2, i1] = contract_ao_wv(den_type, kxc_contract, ao)
    return kxc


def uks_vxc_pot_with_eff(
    den_type: str,
    vxc_eff: np.ndarray,
    ao: np.ndarray,
    weights: np.ndarray,
):
    """Evaluate UKS XC potential (1st order) with vxc_eff.

    Parameters
    ----------
    den_type : str
        The type of electron density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
    vxc_eff : np.ndarray
        The effective XC potential, shape [2, nvar, ngrids].
    ao : np.ndarray
        The basis functions, shape [ncomp, nao, ngrids].
    weights : np.ndarray
        The integration weights, shape [ngrids].

    Returns
    -------
    vxc : np.ndarray
        The XC potential, shape [2, nao, nao].
    """
    nspin, nvar, ngrids = vxc_eff.shape
    assert nspin == 2
    assert weights.shape == (ngrids,)
    assert num_nvar(den_type) == nvar
    assert ao.ndim == 3
    nao = ao.shape[1]
    assert ao.shape[2] == ngrids
    assert ao.shape[0] >= num_ao_comp(den_type)

    if den_type == "LAPL":
        raise NotImplementedError

    vxc = np.zeros((2, nao, nao))
    for s in range(2):
        vxc[s] = contract_ao_wv(den_type, weights * vxc_eff[s], ao)
    return vxc


def uks_fxc_pot_with_eff(
    den_type: str,
    fxc_eff: np.ndarray,
    rho1: np.ndarray,
    ao: np.ndarray,
    weights: np.ndarray,
):
    """Evaluate UKS XC potential (2nd order) with fxc_eff.

    Parameters
    ----------
    den_type : str
        The type of electron density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
    fxc_eff : np.ndarray
        The effective XC kernel, shape [2, nvar, 2, nvar, ngrids].
    rho1 : np.ndarray
        The first-order density response, shape [nset, 2, nvar, ngrids].
    ao : np.ndarray
        The basis functions, shape [ncomp, nao, ngrids].
    weights : np.ndarray
        The integration weights, shape [ngrids].

    Returns
    -------
    fxc : np.ndarray
        The second-order XC potential, shape [nset, 2, nao, nao].
    """
    nset, nspin, nvar, ngrids = rho1.shape
    assert nspin == 2
    assert fxc_eff.shape == (2, nvar, 2, nvar, ngrids)
    assert weights.shape == (ngrids,)
    assert num_nvar(den_type) == nvar
    assert ao.ndim == 3
    nao = ao.shape[1]
    assert ao.shape[2] == ngrids
    assert ao.shape[0] >= num_ao_comp(den_type)

    if den_type == "LAPL":
        raise NotImplementedError

    # Merge spin and var dims: [2, v0, 2, v1, g] -> [2, v0, 2*v1, g]
    fxc_eff_reshape = fxc_eff.reshape(2, nvar, 2 * nvar, ngrids)
    # [nset, 2, v1, g] -> [nset, 2*v1, g]
    rho1_reshape = rho1.reshape(nset, 2 * nvar, ngrids)

    fxc = np.zeros((nset, 2, nao, nao))
    for i in range(nset):
        for s in range(2):
            fxc_contract = weights * (
                fxc_eff_reshape[s] * rho1_reshape[i, None, :, :]
            ).sum(axis=-2)
            fxc[i, s] = contract_ao_wv(den_type, fxc_contract, ao)
    return fxc


def uks_kxc_pot_with_eff(
    den_type: str,
    kxc_eff: np.ndarray,
    rho1: np.ndarray,
    rho2: np.ndarray,
    ao: np.ndarray,
    weights: np.ndarray,
):
    """Evaluate UKS XC potential (3rd order) with kxc_eff.

    Parameters
    ----------
    den_type : str
        The type of electron density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
    kxc_eff : np.ndarray
        The effective XC kernel, shape [2, nvar, 2, nvar, 2, nvar, ngrids].
    rho1 : np.ndarray
        The first-order density response, shape [nset1, 2, nvar, ngrids].
    rho2 : np.ndarray
        The second-order density response, shape [nset2, 2, nvar, ngrids].
    ao : np.ndarray
        The basis functions, shape [ncomp, nao, ngrids].
    weights : np.ndarray
        The integration weights, shape [ngrids].

    Returns
    -------
    kxc : np.ndarray
        The third-order XC potential, shape [nset2, nset1, 2, nao, nao].
    """
    nset1, nspin, nvar, ngrids = rho1.shape
    nset2 = rho2.shape[0]
    assert nspin == 2
    assert kxc_eff.shape == (2, nvar, 2, nvar, 2, nvar, ngrids)
    assert rho2.shape == (nset2, 2, nvar, ngrids)
    assert weights.shape == (ngrids,)
    assert num_nvar(den_type) == nvar
    assert ao.ndim == 3
    nao = ao.shape[1]
    assert ao.shape[2] == ngrids
    assert ao.shape[0] >= num_ao_comp(den_type)

    if den_type == "LAPL":
        raise NotImplementedError

    # Merge spin and var dims
    kxc_eff_reshape = kxc_eff.reshape(2, nvar, 2 * nvar, 2 * nvar, ngrids)
    rho1_reshape = rho1.reshape(nset1, 2 * nvar, ngrids)
    rho2_reshape = rho2.reshape(nset2, 2 * nvar, ngrids)

    kxc = np.zeros((nset2, nset1, 2, nao, nao))
    for i2 in range(nset2):
        for i1 in range(nset1):
            for s in range(2):
                kxc_contract = weights * (
                    kxc_eff_reshape[s]
                    * rho1_reshape[i1, None, None, :, :]
                    * rho2_reshape[i2, None, :, None, :]
                ).sum(axis=(-3, -2))
                kxc[i2, i1, s] = contract_ao_wv(den_type, kxc_contract, ao)
    return kxc


def contract_ao_wv_bra(den_type: str, wv: np.ndarray, ao: np.ndarray, ao_bra: np.ndarray) -> np.ndarray:
    """Contract AO with wv for RHO/SIGMA/TAU.

    Parameters
    ----------
    den_type : str
        The type of density to compute. Can be "RHO", "SIGMA", or "TAU".
    wv : np.ndarray
        The weight vector, shape [nvar, ngrids].
    ao : np.ndarray
        The AO values, shape [ncomp, nao, ngrids].
    ao_bra : np.ndarray
        The bra-transformed AO values, shape [ncomp, nocc, ngrids].

    Returns
    -------
    contracted : np.ndarray
        The contracted AO, shape [nao, nao].
    """
    nvar, ngrids = wv.shape
    assert num_nvar(den_type) == nvar
    assert ao.ndim == 3
    nao = ao.shape[1]
    assert ao.shape[2] == ngrids
    assert ao.shape[0] >= num_ao_comp(den_type)
    assert ao_bra.ndim == 3
    assert ao_bra.shape[2] == ngrids
    assert ao_bra.shape[0] >= num_ao_comp(den_type)
    nocc = ao_bra.shape[1]

    contracted = np.zeros((nocc, nao))
    # RHO contribution
    contracted += ao_bra[0] * wv[0] @ ao[0].T
    # SIGMA contribution
    if den_type in ("SIGMA", "TAU"):
        contracted += ao_bra[1] * wv[1] @ ao[0].T
        contracted += ao_bra[2] * wv[2] @ ao[0].T
        contracted += ao_bra[3] * wv[3] @ ao[0].T
        contracted += ao_bra[0] * wv[1] @ ao[1].T
        contracted += ao_bra[0] * wv[2] @ ao[2].T
        contracted += ao_bra[0] * wv[3] @ ao[3].T
    # TAU contribution
    if den_type in ("TAU",):
        contracted += 0.5 * ao_bra[1] * wv[4] @ ao[1].T
        contracted += 0.5 * ao_bra[2] * wv[4] @ ao[2].T
        contracted += 0.5 * ao_bra[3] * wv[4] @ ao[3].T
    return contracted


def rks_fxc_pot_with_eff_bra_trans(
    den_type: str,
    fxc_eff: np.ndarray,
    rho1: np.ndarray,
    ao: np.ndarray,
    weights: np.ndarray,
    bra: np.ndarray,
):
    r"""Evaluate XC potential (2nd order) with fxc_eff, with bra transformed.

    Bra usually be occupied orbital coefficient (row-major applied to $\mu$), which can lower the computational cost.

    Parameters
    ----------
    den_type : str
        The type of electron density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
    fxc_eff : np.ndarray
        The effective XC kernel, shape [nvar, nvar, ngrids].
    rho1 : np.ndarray
        The first-order density response, shape [nset, nvar, ngrids].
    ao : np.ndarray
        The basis functions, shape [ncomp, nao, ngrids].
    weights : np.ndarray
        The integration weights, shape [ngrids].
    bra : np.ndarray
        The bra orbital coefficients, shape [nao, nocc].

    Returns
    -------
    fxc : np.ndarray
        The second-order XC potential (bra transformed), shape [nset, nocc, nao].
    """
    nset, nvar, ngrids = rho1.shape
    assert fxc_eff.shape == (nvar, nvar, ngrids)
    assert weights.shape == (ngrids,)
    assert ao.ndim == 3
    nao = ao.shape[1]
    assert ao.shape[2] == ngrids
    assert ao.shape[0] >= num_ao_comp(den_type)
    assert bra.ndim == 2
    assert bra.shape[0] == nao
    nocc = bra.shape[1]

    fxc = np.zeros((nset, nocc, nao))
    ao_bra = bra.T @ ao
    for i in range(nset):
        fxc_contract = weights * (fxc_eff * rho1[i, None, :, :]).sum(axis=-2)
        fxc[i] = contract_ao_wv_bra(den_type, fxc_contract, ao, ao_bra)
    return fxc
