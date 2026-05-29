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
// TestEvalRhoPure — naive vs optimized pure function comparison
// ---------------------------------------------------------------------------

mod test_eval_rho_pure {
    use super::*;
    use rstsr_showcase_dft_grids::numint_matmul::pure_eval_rho::{
        get_rho_from_dm_with_output, get_rho_from_homogeneous_braket_with_output,
    };
    use rstsr_showcase_dft_grids::numint_matmul::pure_eval_rho_naive::{
        get_rho_from_dm_with_output_naive, get_rho_from_homogeneous_braket_with_output_naive,
    };

    fn create_out(h2o: &H2OMolecule, den_type: NIDenType) -> Tsr {
        let device = h2o.rdm1.device().clone();
        rt::asarray((vec![0.0; h2o.ngrids * den_type.num_nvar()], [h2o.ngrids, den_type.num_nvar(), 1], &device))
    }

    fn assert_match(a: &Tsr, b: &Tsr, label: &str, den_type: NIDenType) {
        let diff = (a - b).abs().max();
        assert!(diff < 1e-10, "{}: {:?} naive vs opt max diff = {:.3e}", label, den_type, diff);
    }

    #[rstest]
    fn test_get_rho_from_dm_naive_vs_optimized(h2o: &H2OMolecule) {
        let ni_obj = h2o.build_ni_obj();
        let ao = ni_obj.prepare_ao(2);
        let dm_list = [h2o.rdm1.view()];
        for den_type in [RHO, SIGMA, TAU, LAPL] {
            let mut out_naive = create_out(h2o, den_type);
            let mut out_opt = create_out(h2o, den_type);
            get_rho_from_dm_with_output_naive(ao.view(), &dm_list, den_type, out_naive.view_mut()).unwrap();
            get_rho_from_dm_with_output(ao.view(), &dm_list, den_type, out_opt.view_mut(), ni_obj.nchunk).unwrap();
            assert_match(&out_naive, &out_opt, "get_rho_from_dm", den_type);
        }
    }

    #[rstest]
    fn test_get_rho_from_homogeneous_braket_naive_vs_optimized(h2o: &H2OMolecule) {
        let ni_obj = h2o.build_ni_obj();
        let ao = ni_obj.prepare_ao(2);
        let bra = h2o.bra_list();
        let bra_views = [bra[0].view()];
        for den_type in [RHO, SIGMA, TAU, LAPL] {
            let mut out_naive = create_out(h2o, den_type);
            let mut out_opt = create_out(h2o, den_type);
            get_rho_from_homogeneous_braket_with_output_naive(ao.view(), &bra_views, den_type, out_naive.view_mut())
                .unwrap();
            get_rho_from_homogeneous_braket_with_output(
                ao.view(),
                &bra_views,
                den_type,
                out_opt.view_mut(),
                ni_obj.nchunk,
            )
            .unwrap();
            assert_match(&out_naive, &out_opt, "get_rho_from_homogeneous_braket", den_type);
        }
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
