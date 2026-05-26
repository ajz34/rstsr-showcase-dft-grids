import numpy as np
from pynimatmul.flags import num_ao_comp, num_nvar


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
    nao = ao.shape[1]
    assert ao.shape[2] == ngrids
    assert ao.shape[0] >= num_ao_comp(den_type)

    if den_type == "LAPL":
        raise NotImplementedError

    vxc = np.zeros((nao, nao))
    # RHO contribution

    vxc += 0.5 * ao[0] * (weights * vxc_eff[0]) @ ao[0].T
    # SIGMA contribution
    if den_type in ("SIGMA", "TAU"):
        vxc += ao[1] * (weights * vxc_eff[1]) @ ao[0].T
        vxc += ao[2] * (weights * vxc_eff[2]) @ ao[0].T
        vxc += ao[3] * (weights * vxc_eff[3]) @ ao[0].T
    # TAU contribution
    if den_type in ("TAU",):
        wv_4 = weights * vxc_eff[4]
        vxc += 0.25 * ao[1] * wv_4 @ ao[1].T
        vxc += 0.25 * ao[2] * wv_4 @ ao[2].T
        vxc += 0.25 * ao[3] * wv_4 @ ao[3].T
    vxc += vxc.swapaxes(-1, -2)
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
        fxc[i] += 0.5 * ao[0] * fxc_contract[0] @ ao[0].T
        if den_type in ("SIGMA", "TAU"):
            fxc[i] += ao[1] * fxc_contract[1] @ ao[0].T
            fxc[i] += ao[2] * fxc_contract[2] @ ao[0].T
            fxc[i] += ao[3] * fxc_contract[3] @ ao[0].T
        if den_type in ("TAU",):
            fxc[i] += 0.25 * ao[1] * fxc_contract[4] @ ao[1].T
            fxc[i] += 0.25 * ao[2] * fxc_contract[4] @ ao[2].T
            fxc[i] += 0.25 * ao[3] * fxc_contract[4] @ ao[3].T
    fxc += fxc.swapaxes(-1, -2)
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
            kxc[i2, i1] += 0.5 * ao[0] * kxc_contract[0] @ ao[0].T
            if den_type in ("SIGMA", "TAU"):
                kxc[i2, i1] += ao[1] * kxc_contract[1] @ ao[0].T
                kxc[i2, i1] += ao[2] * kxc_contract[2] @ ao[0].T
                kxc[i2, i1] += ao[3] * kxc_contract[3] @ ao[0].T
            if den_type in ("TAU",):
                kxc[i2, i1] += 0.25 * ao[1] * kxc_contract[4] @ ao[1].T
                kxc[i2, i1] += 0.25 * ao[2] * kxc_contract[4] @ ao[2].T
                kxc[i2, i1] += 0.25 * ao[3] * kxc_contract[4] @ ao[3].T
    kxc += kxc.swapaxes(-1, -2)
    return kxc
