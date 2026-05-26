from pyscf import gto, dft
import numpy as np
from pynimatmul.flags import num_nvar, num_ao_comp


def get_rho_from_dm_with_output(
    ao: np.ndarray,
    dm_list: list[np.ndarray],
    den_type: str,
    out: np.ndarray,
    buf: np.ndarray,
):
    """Get the density from the density matrix with output.

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

    Returns
    -------
    np.ndarray
        The computed density with shape [nset, nvar, ngrid].
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
            out[i, 1:4] = (scr * ao[1:4]).sum(axis=0)
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
