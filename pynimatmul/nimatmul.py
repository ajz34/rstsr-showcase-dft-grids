"""
NumInt proposal for REST, reference implementation in Python.

This implementation uses naive BLAS-3 and not optimized.
"""

import numpy as np
from pyscf import gto
from pynimatmul.flags import num_ao_deriv, num_nvar
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
    uks_fxc_pot_with_eff_bra_trans,
    uks_kxc_pot_with_eff,
    uks_vxc_pot_with_eff,
    rks_fxc_pot_with_eff_bra_trans,
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
        """Get AO values with caching.

        Parameters
        ----------
        deriv : int
            The order of derivatives. 0 for AO values, 1 for gradients, 2 for hessians.

        Returns
        -------
        ao : np.ndarray
            The AO values with shape [ncomp, nao, ngrids].
        """
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
        """Evaluate density from density matrices.

        Parameters
        ----------
        dm_list : list[np.ndarray]
            The list of density matrices, each with shape [nao, nao].
        den_type : str
            The type of density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".

        Returns
        -------
        out : np.ndarray
            The computed density, shape [nset, nvar, ngrids].
        """
        ao = self.get_cached_ao(num_ao_deriv(den_type))

        ngrids = ao.shape[2]
        nao = ao.shape[1]
        for dm in dm_list:
            assert dm.ndim == 2
            assert dm.shape == (nao, nao)
        nset = len(dm_list)

        out_shape = (nset, num_nvar(den_type), ngrids)
        out = np.zeros(out_shape)
        return get_rho_from_dm_with_output(ao, dm_list, den_type, out)

    def get_rho_from_homogeneous_braket(
        self, bra_list: list[np.ndarray], den_type: str
    ) -> np.ndarray:
        """Evaluate density from orbital coefficients where bra and ket are the same.

        Parameters
        ----------
        bra_list : list[np.ndarray]
            Orbital coefficient matrices, each with shape [nao, nocc].
        den_type : str
            The type of density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".

        Returns
        -------
        out : np.ndarray
            The computed density, shape [nset, nvar, ngrids].
        """
        ao = self.get_cached_ao(num_ao_deriv(den_type))

        ngrids = ao.shape[2]
        nao = ao.shape[1]
        for bra in bra_list:
            assert bra.ndim == 2
            assert bra.shape[0] == nao
        nset = len(bra_list)

        out_shape = (nset, num_nvar(den_type), ngrids)
        out = np.zeros(out_shape)
        return get_rho_from_homogeneous_braket_with_output(ao, bra_list, den_type, out)

    def get_rho_from_one_bra_mult_ket(
        self, bra: np.ndarray, ket_list: list[np.ndarray], den_type: str
    ) -> np.ndarray:
        """Evaluate density from one shared bra and multiple kets.

        Parameters
        ----------
        bra : np.ndarray
            Shared orbital coefficient matrix with shape [nao, nocc].
        ket_list : list[np.ndarray]
            Orbital coefficient matrices, each with shape [nao, nocc].
        den_type : str
            The type of density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".

        Returns
        -------
        out : np.ndarray
            The computed density, shape [nset, nvar, ngrids].
        """
        ao = self.get_cached_ao(num_ao_deriv(den_type))

        ngrids = ao.shape[2]
        nao = ao.shape[1]
        assert bra.ndim == 2
        assert bra.shape[0] == nao
        nocc = bra.shape[1]
        for ket in ket_list:
            assert ket.ndim == 2
            assert ket.shape[0] == nao
            assert ket.shape[1] == nocc
        nset = len(ket_list)

        out_shape = (nset, num_nvar(den_type), ngrids)
        out = np.zeros(out_shape)
        return get_rho_from_one_bra_mult_ket_with_output(ao, bra, ket_list, den_type, out)

    def get_rho_from_mult_bra_mult_ket(
        self, bra_list: list[np.ndarray], ket_list: list[np.ndarray], den_type: str
    ) -> np.ndarray:
        """Evaluate density from multiple bra-ket pairs.

        Parameters
        ----------
        bra_list : list[np.ndarray]
            Orbital coefficient matrices for bra, each with shape [nao, nocc_i].
        ket_list : list[np.ndarray]
            Orbital coefficient matrices for ket, each with shape [nao, nocc_i].
        den_type : str
            The type of density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".

        Returns
        -------
        out : np.ndarray
            The computed density, shape [nset, nvar, ngrids].
        """
        ao = self.get_cached_ao(num_ao_deriv(den_type))

        ngrids = ao.shape[2]
        nao = ao.shape[1]
        for bra, ket in zip(bra_list, ket_list):
            assert bra.ndim == 2
            assert ket.ndim == 2
            assert bra.shape[0] == nao
            assert ket.shape[0] == nao
            assert bra.shape[1] == ket.shape[1]
        nset = len(bra_list)

        out_shape = (nset, num_nvar(den_type), ngrids)
        out = np.zeros(out_shape)
        return get_rho_from_mult_bra_mult_ket_with_output(
            ao, bra_list, ket_list, den_type, out
        )

    def get_vxc_pot_with_eff(
        self, vxc_eff: np.ndarray, den_type: str, spin: int
    ) -> np.ndarray:
        """Evaluate XC potential (1st order) with vxc_eff.

        Parameters
        ----------
        vxc_eff : np.ndarray
            The effective XC potential, shape [nvar, ngrids] for RKS, [2, nvar, ngrids] for UKS.
        den_type : str
            The type of density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
        spin : int
            0 for RKS, 1 for UKS.

        Returns
        -------
        vxc : np.ndarray
            The XC potential, shape [nao, nao] for RKS, [2, nao, nao] for UKS.
        """
        ao = self.get_cached_ao(num_ao_deriv(den_type))
        f = rks_vxc_pot_with_eff if spin == 0 else uks_vxc_pot_with_eff
        return f(den_type, vxc_eff, ao, self.weights)

    def get_fxc_pot_with_eff(
        self, fxc_eff: np.ndarray, rho1: np.ndarray, den_type: str, spin: int
    ) -> np.ndarray:
        """Evaluate XC potential (2nd order) with fxc_eff.

        Parameters
        ----------
        fxc_eff : np.ndarray
            The effective XC kernel, shape [nvar, nvar, ngrids] for RKS, [2, nvar, 2, nvar, ngrids] for UKS.
        rho1 : np.ndarray
            The first-order density response, shape [nset, nvar, ngrids] for RKS, [nset, 2, nvar, ngrids] for UKS.
        den_type : str
            The type of density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
        spin : int
            0 for RKS, 1 for UKS.

        Returns
        -------
        fxc : np.ndarray
            The second-order XC potential, shape [nset, nao, nao] for RKS, [nset, 2, nao, nao] for UKS.
        """
        ao = self.get_cached_ao(num_ao_deriv(den_type))
        f = rks_fxc_pot_with_eff if spin == 0 else uks_fxc_pot_with_eff
        return f(den_type, fxc_eff, rho1, ao, self.weights)

    def get_kxc_pot_with_eff(
        self,
        kxc_eff: np.ndarray,
        rho1: np.ndarray,
        rho2: np.ndarray,
        den_type: str,
        spin: int,
    ) -> np.ndarray:
        """Evaluate XC potential (3rd order) with kxc_eff.

        Parameters
        ----------
        kxc_eff : np.ndarray
            The effective XC kernel, shape [nvar, nvar, nvar, ngrids] for RKS,
            [2, nvar, 2, nvar, 2, nvar, ngrids] for UKS.
        rho1 : np.ndarray
            The first-order density response, shape [nset1, nvar, ngrids] for RKS,
            [nset1, 2, nvar, ngrids] for UKS.
        rho2 : np.ndarray
            The second-order density response, shape [nset2, nvar, ngrids] for RKS,
            [nset2, 2, nvar, ngrids] for UKS.
        den_type : str
            The type of density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
        spin : int
            0 for RKS, 1 for UKS.

        Returns
        -------
        kxc : np.ndarray
            The third-order XC potential, shape [nset2, nset1, nao, nao] for RKS,
            [nset2, nset1, 2, nao, nao] for UKS.
        """
        ao = self.get_cached_ao(num_ao_deriv(den_type))
        f = rks_kxc_pot_with_eff if spin == 0 else uks_kxc_pot_with_eff
        return f(den_type, kxc_eff, rho1, rho2, ao, self.weights)

    def get_fxc_pot_with_eff_bra_trans(
        self,
        fxc_eff: np.ndarray,
        rho1: np.ndarray,
        bra: np.ndarray,
        den_type: str,
        spin: int,
    ):
        """Evaluate XC potential (2nd order) with fxc_eff, with bra transformed.

        Parameters
        ----------
        fxc_eff : np.ndarray
            The effective XC kernel, shape [nvar, nvar, ngrids] for RKS,
            [2, nvar, 2, nvar, ngrids] for UKS.
        rho1 : np.ndarray
            The first-order density response, shape [nset, nvar, ngrids] for RKS,
            [nset, 2, nvar, ngrids] for UKS.
        bra : np.ndarray or list[np.ndarray]
            The bra orbital coefficients. For RKS: shape [nao, nocc].
            For UKS: list of two arrays [nao, nocc_alpha] and [nao, nocc_beta].
        den_type : str
            The type of density to compute. Can be "RHO", "SIGMA", "TAU", or "LAPL".
        spin : int
            0 for RKS, 1 for UKS.

        Returns
        -------
        fxc : np.ndarray or list[np.ndarray]
            The second-order XC potential (bra transformed).
            For RKS: shape [nset, nocc, nao].
            For UKS: list of two arrays [nset, nocc_alpha, nao] and [nset, nocc_beta, nao].
        """
        ao = self.get_cached_ao(num_ao_deriv(den_type))
        f = (
            rks_fxc_pot_with_eff_bra_trans
            if spin == 0
            else uks_fxc_pot_with_eff_bra_trans
        )
        return f(den_type, fxc_eff, rho1, ao, self.weights, bra)
