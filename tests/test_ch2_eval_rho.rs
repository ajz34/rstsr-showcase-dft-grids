mod test_util;

use rstest::*;
use rstsr::prelude::*;
use rstsr_showcase_dft_grids::prelude::*;
use test_util::*;

use NIDenType::*;

// ---------------------------------------------------------------------------
// Fixtures (≡ Python setUpModule)
// ---------------------------------------------------------------------------

#[fixture]
#[once]
fn ch2() -> Ch2Molecule {
    Ch2Molecule::load()
}

// ---------------------------------------------------------------------------
// TestGetRhoFromDM
// ---------------------------------------------------------------------------

mod test_get_rho_from_dm {
    use super::*;

    #[rstest]
    fn test_rho(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let dm_list = [ch2.rdm1.i((.., .., 0)), ch2.rdm1.i((.., .., 1))];
        let out = ni_obj.make_rho_from_dm(&dm_list, RHO).unwrap();
        assert_eq!(out.shape(), &[ch2.ngrids, 1, 2]);
        fp_assert_eq!(out.view(), 90.1600267407401, 1e-6);
    }

    #[rstest]
    fn test_sigma(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let dm_list = [ch2.rdm1.i((.., .., 0)), ch2.rdm1.i((.., .., 1))];
        let out = ni_obj.make_rho_from_dm(&dm_list, SIGMA).unwrap();
        assert_eq!(out.shape(), &[ch2.ngrids, 4, 2]);
        fp_assert_eq!(out.view(), 369.8178428546338, 1e-6);
    }

    #[rstest]
    fn test_tau(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let dm_list = [ch2.rdm1.i((.., .., 0)), ch2.rdm1.i((.., .., 1))];
        let out = ni_obj.make_rho_from_dm(&dm_list, TAU).unwrap();
        assert_eq!(out.shape(), &[ch2.ngrids, 5, 2]);
        fp_assert_eq!(out.view(), 5587.859016487346, 1e-6);
    }

    #[rstest]
    fn test_lapl(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let dm_list = [ch2.rdm1.i((.., .., 0)), ch2.rdm1.i((.., .., 1))];
        let out = ni_obj.make_rho_from_dm(&dm_list, LAPL).unwrap();
        assert_eq!(out.shape(), &[ch2.ngrids, 6, 2]);
        fp_assert_eq!(out.i((.., ..5, ..)), 5587.859016487346, 1e-6);
        fp_assert_eq!(out.i((.., 5, ..)), -783527.5368333755, 1e-4);
    }
}

// ---------------------------------------------------------------------------
// TestGetRhoFromHomogeneousBraket
// ---------------------------------------------------------------------------

mod test_get_rho_from_homogeneous_braket {
    use super::*;

    #[rstest]
    fn test_rho(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let bra_list = ch2.bra_list();
        let bra_views = [bra_list[0].view(), bra_list[1].view()];
        let out = ni_obj.make_rho_from_homogeneous_braket(&bra_views, RHO).unwrap();
        assert_eq!(out.shape(), &[ch2.ngrids, 1, 2]);
        fp_assert_eq!(out.view(), 90.1600267407401, 1e-6);
    }

    #[rstest]
    fn test_sigma(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let bra_list = ch2.bra_list();
        let bra_views = [bra_list[0].view(), bra_list[1].view()];
        let out = ni_obj.make_rho_from_homogeneous_braket(&bra_views, SIGMA).unwrap();
        assert_eq!(out.shape(), &[ch2.ngrids, 4, 2]);
        fp_assert_eq!(out.view(), 369.8178428546338, 1e-6);
    }

    #[rstest]
    fn test_tau(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let bra_list = ch2.bra_list();
        let bra_views = [bra_list[0].view(), bra_list[1].view()];
        let out = ni_obj.make_rho_from_homogeneous_braket(&bra_views, TAU).unwrap();
        assert_eq!(out.shape(), &[ch2.ngrids, 5, 2]);
        fp_assert_eq!(out.view(), 5587.859016487346, 1e-6);
    }

    #[rstest]
    fn test_lapl(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let bra_list = ch2.bra_list();
        let bra_views = [bra_list[0].view(), bra_list[1].view()];
        let out = ni_obj.make_rho_from_homogeneous_braket(&bra_views, LAPL).unwrap();
        assert_eq!(out.shape(), &[ch2.ngrids, 6, 2]);
        fp_assert_eq!(out.i((.., ..5, ..)), 5587.859016487346, 1e-6);
        fp_assert_eq!(out.i((.., 5.., ..)), -783527.5368333755, 1e-4);
    }
}

// ---------------------------------------------------------------------------
// TestGetRhoFromOneBraMultKet
// ---------------------------------------------------------------------------

mod test_get_rho_from_one_bra_mult_ket {
    use super::*;

    fn get_bra_ket(ch2: &Ch2Molecule) -> (Tsr, [Tsr; 2]) {
        let bra = ch2.mo_coeff.i((0, .., ..4)).to_owned().into_contig(ColMajor);
        let ket_a = ch2.mo_coeff.i((0, .., ..4)).to_owned().into_contig(ColMajor);
        let ket_b = ch2.mo_coeff.i((1, .., ..4)).to_owned().into_contig(ColMajor);
        (bra, [ket_a, ket_b])
    }

    #[rstest]
    fn test_rho(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let (bra, ket_list) = get_bra_ket(ch2);
        let ket_views = [ket_list[0].view(), ket_list[1].view()];
        let out = ni_obj.make_rho_from_one_bra_mult_ket(bra.view(), &ket_views, RHO).unwrap();
        assert_eq!(out.shape(), &[ch2.ngrids, 1, 2]);
        fp_assert_eq!(out.view(), 90.19801284927239, 1e-6);
    }

    #[rstest]
    fn test_cross_check(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let (bra, ket_list) = get_bra_ket(ch2);
        let ket_views = [ket_list[0].view(), ket_list[1].view()];
        let dm_a = bra.matmul(&ket_list[0].t());
        let dm_a_sym = (&dm_a + &dm_a.t()) * 0.5;
        let dm_b = bra.matmul(&ket_list[1].t());
        let dm_b_sym = (&dm_b + &dm_b.t()) * 0.5;
        for den_type in [RHO, SIGMA, TAU, LAPL] {
            let out = ni_obj.make_rho_from_one_bra_mult_ket(bra.view(), &ket_views, den_type).unwrap();
            let ref_a = ni_obj.make_rho_from_dm(&[dm_a_sym.view()], den_type).unwrap();
            let ref_b = ni_obj.make_rho_from_dm(&[dm_b_sym.view()], den_type).unwrap();
            let diff_a = (&out.i((.., .., 0)) - &ref_a.i((.., .., 0))).abs().max();
            let diff_b = (&out.i((.., .., 1)) - &ref_b.i((.., .., 0))).abs().max();
            assert!(diff_a < 1e-8, "Mismatch for alpha set, {:?}: max diff = {}", den_type, diff_a);
            assert!(diff_b < 1e-8, "Mismatch for beta set, {:?}: max diff = {}", den_type, diff_b);
        }
    }
}

// ---------------------------------------------------------------------------
// TestGetRhoFromMultBraMultKet
// ---------------------------------------------------------------------------

mod test_get_rho_from_mult_bra_mult_ket {
    use super::*;

    fn get_bra_ket_lists(ch2: &Ch2Molecule) -> ([Tsr; 2], [Tsr; 2]) {
        let bra_a = ch2.mo_coeff.i((0, .., ..4)).to_owned().into_contig(ColMajor);
        let bra_b = ch2.mo_coeff.i((1, .., ..4)).to_owned().into_contig(ColMajor);
        let ket_a = ch2.mo_coeff.i((0, .., ..4)).to_owned().into_contig(ColMajor);
        let ket_b = ch2.mo_coeff.i((1, .., ..4)).to_owned().into_contig(ColMajor);
        ([bra_a, bra_b], [ket_a, ket_b])
    }

    #[rstest]
    fn test_cross_check(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let (bra_list, ket_list) = get_bra_ket_lists(ch2);
        let bra_views = [bra_list[0].view(), bra_list[1].view()];
        let ket_views = [ket_list[0].view(), ket_list[1].view()];
        let dm_a = bra_list[0].matmul(&ket_list[0].t());
        let dm_a_sym = (&dm_a + &dm_a.t()) * 0.5;
        let dm_b = bra_list[1].matmul(&ket_list[1].t());
        let dm_b_sym = (&dm_b + &dm_b.t()) * 0.5;
        for den_type in [RHO, SIGMA, TAU, LAPL] {
            let out = ni_obj.make_rho_from_mult_bra_mult_ket(&bra_views, &ket_views, den_type).unwrap();
            assert_eq!(out.shape(), &[ch2.ngrids, den_type.num_nvar(), 2]);
            let ref_a = ni_obj.make_rho_from_dm(&[dm_a_sym.view()], den_type).unwrap();
            let ref_b = ni_obj.make_rho_from_dm(&[dm_b_sym.view()], den_type).unwrap();
            let diff_a = (&out.i((.., .., 0)) - &ref_a.i((.., .., 0))).abs().max();
            let diff_b = (&out.i((.., .., 1)) - &ref_b.i((.., .., 0))).abs().max();
            assert!(diff_a < 1e-8, "Mismatch for alpha set, {:?}: max diff = {}", den_type, diff_a);
            assert!(diff_b < 1e-8, "Mismatch for beta set, {:?}: max diff = {}", den_type, diff_b);
        }
    }
}
