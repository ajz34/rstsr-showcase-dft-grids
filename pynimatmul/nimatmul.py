"""
NumInt proposal for REST, reference implementation in Python.

This implementation uses naive BLAS-3 and not optimized.
"""

from pyscf import gto, dft
import numpy as np
from pynimatmul.flags import num_ao_deriv, num_nvar, num_ao_comp
from pynimatmul.pure_eval_rho import (
    get_rho_from_dm_with_output,
    get_rho_from_homogeneous_braket_with_output,
    get_rho_from_one_bra_mult_ket_with_output,
    get_rho_from_mult_bra_mult_ket_with_output,
)
from pynimatmul.pure_xcpot import (
    rks_fxc_pot_with_eff,
    rks_kxc_pot_with_eff,
    rks_vxc_pot_with_eff,
    uks_fxc_pot_with_eff,
    uks_kxc_pot_with_eff,
    uks_vxc_pot_with_eff,
)


class NIMatmul:
    mol: gto.Mole
    coords: np.ndarray
    weights: np.ndarray
    cache_tensor: dict[str, np.ndarray]

    def __init__(
        self,
        mol: gto.Mole,
        coords: np.ndarray,
        weights: np.ndarray,
    ):
        assert coords.ndim == 2
        assert weights.ndim == 1
        assert coords.shape[0] == weights.shape[0]
        assert coords.shape[1] == 3

        self.mol = mol
        self.coords = coords
        self.weights = weights
        self.cache_tensor = {}

    def get_ao(self, deriv: int) -> np.ndarray:
        """Get the AO values.

        Arguments:
            deriv: int, the order of derivatives.
            - 0 for AO values (RHO),
            - 1 for gradients (SIGMA, TAU)
            - 2 for hessians (LAPL)

        Returns:
            AO with shape [ncomp, nao, ngrid]
        """
        eval_name = f"GTOval_sph_deriv{deriv}"
        grids = self.mol.eval_gto(eval_name, self.coords)
        # convention change from PySCF
        # - deriv = 0 still returns 3-dim tensor
        # - use [ncomp, nao, ngrid] instead of [ncomp, ngrid, nao] (row-major)
        if deriv == 0:
            grids = grids[None, :, :]
        return grids.transpose(0, 2, 1)

    def get_cached_ao(self, deriv: int):
        AO_DERIV_DIM = [1, 4, 10, 20, 35]
        assert deriv < len(AO_DERIV_DIM)
        # determine maximum cached derivative level
        max_cached_deriv = (
            max([int(key.split("deriv")[1]) for key in self.cache_tensor.keys()])
            if self.cache_tensor
            else -1
        )
        if max_cached_deriv >= deriv:
            filter_str = f"GTOval_sph_deriv{max_cached_deriv}"
            return self.cache_tensor[filter_str][: AO_DERIV_DIM[deriv], :, :]
        filter_str = f"GTOval_sph_deriv{deriv}"
        self.cache_tensor[filter_str] = self.get_ao(deriv)
        return self.cache_tensor[filter_str]

    def get_rho_from_dm(self, dm_list: list[np.ndarray], den_type: str) -> np.ndarray:
        ao = self.get_cached_ao(num_ao_deriv(den_type))

        ngrids = ao.shape[2]
        nao = ao.shape[1]
        for dm in dm_list:
            assert dm.ndim == 2
            assert dm.shape == (nao, nao)
        nset = len(dm_list)

        out_shape = (nset, num_nvar(den_type), ngrids)
        out = np.zeros(out_shape)
        buf = np.empty(ngrids * nao)
        return get_rho_from_dm_with_output(ao, dm_list, den_type, out, buf)

    def get_rho_from_homogeneous_braket(
        self, bra_list: list[np.ndarray], den_type: str
    ) -> np.ndarray:
        ao = self.get_cached_ao(num_ao_deriv(den_type))

        ngrid = ao.shape[2]
        nao = ao.shape[1]
        for bra in bra_list:
            assert bra.ndim == 2
            assert bra.shape[0] == nao
        nocc_max = max(bra.shape[1] for bra in bra_list) if bra_list else 0
        nset = len(bra_list)

        out_shape = (nset, num_nvar(den_type), ngrid)
        out = np.zeros(out_shape)
        buf = np.empty(2 * ngrid * nocc_max)
        return get_rho_from_homogeneous_braket_with_output(
            ao, bra_list, den_type, out, buf
        )

    def get_rho_from_one_bra_mult_ket(
        self, bra: np.ndarray, ket_list: list[np.ndarray], den_type: str
    ) -> np.ndarray:
        ao = self.get_cached_ao(num_ao_deriv(den_type))

        ngrid = ao.shape[2]
        nao = ao.shape[1]
        assert bra.ndim == 2
        assert bra.shape[0] == nao
        nocc = bra.shape[1]
        for ket in ket_list:
            assert ket.ndim == 2
            assert ket.shape[0] == nao
            assert ket.shape[1] == nocc
        nset = len(ket_list)

        out_shape = (nset, num_nvar(den_type), ngrid)
        out = np.zeros(out_shape)
        buf = np.empty(3 * ngrid * nocc)
        return get_rho_from_one_bra_mult_ket_with_output(
            ao, bra, ket_list, den_type, out, buf
        )

    def get_rho_from_mult_bra_mult_ket(
        self, bra_list: list[np.ndarray], ket_list: list[np.ndarray], den_type: str
    ) -> np.ndarray:
        ao = self.get_cached_ao(num_ao_deriv(den_type))

        ngrid = ao.shape[2]
        nao = ao.shape[1]
        for bra, ket in zip(bra_list, ket_list):
            assert bra.ndim == 2
            assert ket.ndim == 2
            assert bra.shape[0] == nao
            assert ket.shape[0] == nao
            assert bra.shape[1] == ket.shape[1]
        nocc_max = max(bra.shape[1] for bra in bra_list) if bra_list else 0
        nset = len(bra_list)

        out_shape = (nset, num_nvar(den_type), ngrid)
        out = np.zeros(out_shape)
        buf = np.empty(3 * ngrid * nocc_max)
        return get_rho_from_mult_bra_mult_ket_with_output(
            ao, bra_list, ket_list, den_type, out, buf
        )

    def get_vxc_pot_with_eff(
        self, vxc_eff: np.ndarray, den_type: str, spin: int
    ) -> np.ndarray:
        ao = self.get_cached_ao(num_ao_deriv(den_type))
        if spin == 0:
            return rks_vxc_pot_with_eff(den_type, vxc_eff, ao, self.weights)
        else:
            return uks_vxc_pot_with_eff(den_type, vxc_eff, ao, self.weights)

    def get_fxc_pot_with_eff(
        self, fxc_eff: np.ndarray, rho1: np.ndarray, den_type: str, spin: int
    ) -> np.ndarray:
        ao = self.get_cached_ao(num_ao_deriv(den_type))
        if spin == 0:
            return rks_fxc_pot_with_eff(den_type, fxc_eff, rho1, ao, self.weights)
        else:
            return uks_fxc_pot_with_eff(den_type, fxc_eff, rho1, ao, self.weights)

    def get_kxc_pot_with_eff(
        self,
        kxc_eff: np.ndarray,
        rho1: np.ndarray,
        rho2: np.ndarray,
        den_type: str,
        spin: int,
    ) -> np.ndarray:
        ao = self.get_cached_ao(num_ao_deriv(den_type))
        if spin == 0:
            return rks_kxc_pot_with_eff(den_type, kxc_eff, rho1, rho2, ao, self.weights)
        else:
            return uks_kxc_pot_with_eff(den_type, kxc_eff, rho1, rho2, ao, self.weights)
