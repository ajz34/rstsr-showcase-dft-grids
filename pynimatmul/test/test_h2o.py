import unittest
from pyscf import gto, dft, lib
from pynimatmul.nimatmul import NIMatmul
import numpy as np


def setUpModule():
    global mol, mf, mo_coeff, mo_occ, rdm1, coords, weights, ngrids
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="def2-TZVP").build()
    mf = dft.RKS(mol, xc="TPSS0").run()
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    rdm1 = mf.make_rdm1()
    coords = mf.grids.coords
    weights = mf.grids.weights
    ngrids = coords.shape[0]

    global grids, ni
    grids = mf.grids
    ni = dft.numint.NumInt()


class TestGetRhoFromDM(unittest.TestCase):
    def test_rho(self):
        ni_obj = NIMatmul(mol, coords, weights)
        rho = ni_obj.get_rho_from_dm([rdm1], "RHO")
        self.assertEqual(rho.shape, (1, 1, ngrids))
        self.assertAlmostEqual(lib.fp(rho), -438.0303348067822, places=6)

    def test_sigma(self):
        ni_obj = NIMatmul(mol, coords, weights)
        sigma = ni_obj.get_rho_from_dm([rdm1], "SIGMA")
        self.assertEqual(sigma.shape, (1, 4, ngrids))
        self.assertAlmostEqual(lib.fp(sigma), 25704.14480085447, places=4)

    def test_tau(self):
        ni_obj = NIMatmul(mol, coords, weights)
        tau = ni_obj.get_rho_from_dm([rdm1], "TAU")
        self.assertEqual(tau.shape, (1, 5, ngrids))
        self.assertAlmostEqual(lib.fp(tau), 17140.30079158995, places=4)

    def test_lapl(self):
        ni_obj = NIMatmul(mol, coords, weights)
        lapl = ni_obj.get_rho_from_dm([rdm1], "LAPL")
        self.assertEqual(lapl.shape, (1, 6, ngrids))
        self.assertAlmostEqual(lib.fp(lapl[:, :5, :]), 17140.30079158995, places=4)
        self.assertAlmostEqual(lib.fp(lapl[:, 5, :]), 2470300.187572363, places=1)


class TestGetRhoFromHomogeneousBraket(unittest.TestCase):
    def _get_bra(self):
        occ_mask = mo_occ > 0
        return mo_coeff[:, occ_mask] * np.sqrt(mo_occ[occ_mask])

    def test_rho(self):
        bra = self._get_bra()
        ni_obj = NIMatmul(mol, coords, weights)
        out = ni_obj.get_rho_from_homogeneous_braket([bra], "RHO")
        ref = ni_obj.get_rho_from_dm([rdm1], "RHO")
        self.assertEqual(out.shape, (1, 1, ngrids))
        self.assertAlmostEqual(np.abs(out - ref).max(), 0, places=8)

    def test_sigma(self):
        bra = self._get_bra()
        ni_obj = NIMatmul(mol, coords, weights)
        out = ni_obj.get_rho_from_homogeneous_braket([bra], "SIGMA")
        ref = ni_obj.get_rho_from_dm([rdm1], "SIGMA")
        self.assertEqual(out.shape, (1, 4, ngrids))
        self.assertAlmostEqual(np.abs(out - ref).max(), 0, places=8)

    def test_tau(self):
        bra = self._get_bra()
        ni_obj = NIMatmul(mol, coords, weights)
        out = ni_obj.get_rho_from_homogeneous_braket([bra], "TAU")
        ref = ni_obj.get_rho_from_dm([rdm1], "TAU")
        self.assertEqual(out.shape, (1, 5, ngrids))
        self.assertAlmostEqual(np.abs(out - ref).max(), 0, places=8)

    def test_lapl(self):
        bra = self._get_bra()
        ni_obj = NIMatmul(mol, coords, weights)
        out = ni_obj.get_rho_from_homogeneous_braket([bra], "LAPL")
        ref = ni_obj.get_rho_from_dm([rdm1], "LAPL")
        self.assertEqual(out.shape, (1, 6, ngrids))
        self.assertAlmostEqual(np.abs(out - ref).max(), 0, places=6)
