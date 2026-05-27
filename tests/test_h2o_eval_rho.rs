mod test_util;

use rstest::*;
use rstsr_showcase_dft_grids::prelude::*;
use test_util::*;

use NIDenType::*;

// ---------------------------------------------------------------------------
// Fixtures (≡ Python setUpModule)
// ---------------------------------------------------------------------------

#[fixture]
#[once]
fn h2o() -> H2OMolecule {
    H2OMolecule::load()
}

// ---------------------------------------------------------------------------
// TestGetRhoFromDM
// ---------------------------------------------------------------------------

mod test_get_rho_from_dm {
    use super::*;

    #[rstest]
    fn test_rho(h2o: &H2OMolecule) {
        let mut ni_obj = h2o.build_ni_obj();
        let out = ni_obj.make_rho_from_dm(&[h2o.rdm1.view()], RHO).unwrap();
        assert_eq!(out.shape(), &[h2o.ngrids, 1, 1]);
        fp_assert_eq!(out.view(), -438.0303348067822, 1e-6);
    }

    #[rstest]
    fn test_sigma(h2o: &H2OMolecule) {
        let mut ni_obj = h2o.build_ni_obj();
        let out = ni_obj.make_rho_from_dm(&[h2o.rdm1.view()], SIGMA).unwrap();
        assert_eq!(out.shape(), &[h2o.ngrids, 4, 1]);
        fp_assert_eq!(out.view(), 25704.14480085445, 1e-6);
    }

    #[rstest]
    fn test_tau(h2o: &H2OMolecule) {
        let mut ni_obj = h2o.build_ni_obj();
        let out = ni_obj.make_rho_from_dm(&[h2o.rdm1.view()], TAU).unwrap();
        assert_eq!(out.shape(), &[h2o.ngrids, 5, 1]);
        fp_assert_eq!(out.view(), 17140.300791589965, 1e-6);
    }

    #[rstest]
    fn test_lapl(h2o: &H2OMolecule) {
        let mut ni_obj = h2o.build_ni_obj();
        let out = ni_obj.make_rho_from_dm(&[h2o.rdm1.view()], LAPL).unwrap();
        assert_eq!(out.shape(), &[h2o.ngrids, 6, 1]);
        fp_assert_eq!(out.i((.., ..5, ..)), 17140.300791589965, 1e-6);
        fp_assert_eq!(out.i((.., 5, ..)), 2470300.1875723703, 1e-4);
    }
}

// ---------------------------------------------------------------------------
// TestGetRhoFromHomogeneousBraket
// ---------------------------------------------------------------------------

mod test_get_rho_from_homogeneous_braket {
    use super::*;

    #[rstest]
    fn test_rho(h2o: &H2OMolecule) {
        let mut ni_obj = h2o.build_ni_obj();
        let bra = h2o.bra_list();
        let out = ni_obj.make_rho_from_homogeneous_braket(&[bra[0].view()], RHO).unwrap();
        assert_eq!(out.shape(), &[h2o.ngrids, 1, 1]);
        fp_assert_eq!(out.view(), -438.0303348067822, 1e-6);
    }

    #[rstest]
    fn test_sigma(h2o: &H2OMolecule) {
        let mut ni_obj = h2o.build_ni_obj();
        let bra = h2o.bra_list();
        let out = ni_obj.make_rho_from_homogeneous_braket(&[bra[0].view()], SIGMA).unwrap();
        assert_eq!(out.shape(), &[h2o.ngrids, 4, 1]);
        fp_assert_eq!(out.view(), 25704.14480085445, 1e-6);
    }

    #[rstest]
    fn test_tau(h2o: &H2OMolecule) {
        let mut ni_obj = h2o.build_ni_obj();
        let bra = h2o.bra_list();
        let out = ni_obj.make_rho_from_homogeneous_braket(&[bra[0].view()], TAU).unwrap();
        assert_eq!(out.shape(), &[h2o.ngrids, 5, 1]);
        fp_assert_eq!(out.view(), 17140.300791589965, 1e-6);
    }

    #[rstest]
    fn test_lapl(h2o: &H2OMolecule) {
        let mut ni_obj = h2o.build_ni_obj();
        let bra = h2o.bra_list();
        let out = ni_obj.make_rho_from_homogeneous_braket(&[bra[0].view()], LAPL).unwrap();
        assert_eq!(out.shape(), &[h2o.ngrids, 6, 1]);
        fp_assert_eq!(out.i((.., ..5, ..)), 17140.300791589965, 1e-6);
        fp_assert_eq!(out.i((.., 5, ..)), 2470300.1875723703, 1e-4);
    }
}
