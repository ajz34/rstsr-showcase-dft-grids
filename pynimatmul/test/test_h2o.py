import unittest
from pyscf import gto, dft, lib
from pynimatmul.nimatmul import NIMatmul


def setUpModule():
    global mol, mf, mo_coeff, mo_occ, rdm1, coords, weights
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="def2-TZVP").build()
    mf = dft.RKS(mol, xc="TPSS0").run()
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    rdm1 = mf.make_rdm1()
    coords = mf.grids.coords
    weights = mf.grids.weights

    global grids, ni
    grids = mf.grids
    ni = dft.numint.NumInt()


class KnownValues(unittest.TestCase):
    def test_get_rho_from_dm(self):
        global mol, mf, mo_coeff, mo_occ, rdm1, coords, weights
        coords = grids.coords
        weights = grids.weights
        ni_obj = NIMatmul(mol, coords, weights)
        rho = ni_obj.get_rho_from_dm([rdm1], "RHO")
        self.assertEqual(rho.shape, (1, 1, coords.shape[0]))
        self.assertAlmostEqual(lib.fp(rho), -438.0303348067822, places=6)
