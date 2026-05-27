mod test_util;

use libxc::prelude::*;
use rstest::*;
use rstsr::prelude::*;
use rstsr_showcase_dft_grids::prelude::*;
use test_util::*;

use LibXCSpin::*;
use NIDenType::*;

type Tsr<T = f64> = Tensor<T, DeviceFaer, IxD>;

// ---------------------------------------------------------------------------
// Fixtures (≡ Python setUpModule + TestXCPot.setUpClass)
// ---------------------------------------------------------------------------

/// Perturbed density matrices for UKS, precomputed once.
pub struct Ch2PerturbedDM {
    pub dm1_flat: Tsr,
    pub dm2_flat: Tsr,
    pub ncomp1: usize,
    pub ncomp2: usize,
}

#[fixture]
#[once]
fn ch2() -> Ch2Molecule {
    Ch2Molecule::load()
}

#[fixture]
#[once]
fn perturbed_dm(ch2: &Ch2Molecule) -> Ch2PerturbedDM {
    let device = ch2.rdm1.device().clone();
    let get_intor = |name: &str| {
        let (out, shape) = ch2.cint().integrate(name, None, None).into();
        rt::asarray((out, shape, &device))
    };
    let int1e_r_giao: Tsr = get_intor("int1e_r") + get_intor("int1e_giao_irjxp");
    let int1e_rr: Tsr = get_intor("int1e_rr");
    let ncomp1 = int1e_r_giao.shape()[int1e_r_giao.ndim() - 1];
    let ncomp2 = int1e_rr.shape()[int1e_rr.ndim() - 1];
    let nao = ch2.rdm1.shape()[0];

    let dm1_raw = int1e_r_giao.i((.., .., None, ..)) * ch2.rdm1.i((.., .., .., None));
    let dm1: Tsr = 0.5 * (&dm1_raw + dm1_raw.swapaxes(0, 1));
    let dm2_raw = int1e_rr.i((.., .., None, ..)) * ch2.rdm1.i((.., .., .., None));
    let dm2: Tsr = 0.5 * (&dm2_raw + dm2_raw.swapaxes(0, 1));

    let dm1_flat: Tsr = dm1.into_shape([nao, nao, 2 * ncomp1]);
    let dm2_flat: Tsr = dm2.into_shape([nao, nao, 2 * ncomp2]);

    Ch2PerturbedDM { dm1_flat, dm2_flat, ncomp1, ncomp2 }
}

// ---------------------------------------------------------------------------
// TestXCPot
// ---------------------------------------------------------------------------

mod test_xcpot {
    use super::*;

    fn dm0_list(ch2: &Ch2Molecule) -> [TsrView<'_>; 2] {
        [ch2.rdm1.i((.., .., 0)), ch2.rdm1.i((.., .., 1))]
    }

    #[rstest]
    fn test_rho(ch2: &Ch2Molecule, perturbed_dm: &Ch2PerturbedDM) {
        let mut ni_obj = ch2.build_ni_obj();
        let rho0 = ni_obj.make_rho_from_dm(&dm0_list(ch2), RHO).unwrap();
        let dm1_list: Vec<_> = perturbed_dm.dm1_flat.axes_iter(-1).collect();
        let dm2_list: Vec<_> = perturbed_dm.dm2_flat.axes_iter(-1).collect();
        let rho1: Tsr =
            ni_obj.make_rho_from_dm(&dm1_list, RHO).unwrap().into_shape([ch2.ngrids, 1, 2, perturbed_dm.ncomp1]);
        let rho2: Tsr =
            ni_obj.make_rho_from_dm(&dm2_list, RHO).unwrap().into_shape([ch2.ngrids, 1, 2, perturbed_dm.ncomp2]);

        let xc_func = LibXCFunctional::from_identifier("LDA_X", Polarized);
        let xc_eff = libxc_eval_eff(&xc_func, rho0.view(), 3, true).unwrap();
        let [exc_eff, vxc_eff, fxc_eff, kxc_eff] = xc_eff.try_into().unwrap();
        let exc = (exc_eff * (rho0.i((.., 0, 0)) + rho0.i((.., 0, 1))) * &ch2.weights).sum();
        let vxc = ni_obj.make_vxc_pot_with_eff(vxc_eff.view(), RHO, 1).unwrap();
        let fxc = ni_obj.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), RHO, 1).unwrap();
        let kxc = ni_obj.make_kxc_pot_with_eff(kxc_eff.view(), rho1.view(), rho2.view(), RHO, 1).unwrap();
        assert!((exc - -4.7040426008).abs() < 1e-6);
        fp_assert_eq!(vxc.view(), -12.7427734694, 1e-6);
        fp_assert_eq!(fxc.view(), -0.2560478462754152, 1e-6);
        fp_assert_eq!(kxc.view(), 0.40388776601225995, 1e-6);
    }

    #[rstest]
    fn test_sigma(ch2: &Ch2Molecule, perturbed_dm: &Ch2PerturbedDM) {
        let mut ni_obj = ch2.build_ni_obj();
        let rho0 = ni_obj.make_rho_from_dm(&dm0_list(ch2), SIGMA).unwrap();
        let dm1_list: Vec<_> = perturbed_dm.dm1_flat.axes_iter(-1).collect();
        let dm2_list: Vec<_> = perturbed_dm.dm2_flat.axes_iter(-1).collect();
        let rho1: Tsr =
            ni_obj.make_rho_from_dm(&dm1_list, SIGMA).unwrap().into_shape([ch2.ngrids, 4, 2, perturbed_dm.ncomp1]);
        let rho2: Tsr =
            ni_obj.make_rho_from_dm(&dm2_list, SIGMA).unwrap().into_shape([ch2.ngrids, 4, 2, perturbed_dm.ncomp2]);

        let xc_func = LibXCFunctional::from_identifier("GGA_X_PBE", Polarized);
        let xc_eff = libxc_eval_eff(&xc_func, rho0.view(), 3, true).unwrap();
        let [exc_eff, vxc_eff, fxc_eff, kxc_eff] = xc_eff.try_into().unwrap();
        let exc = (exc_eff * (rho0.i((.., 0, 0)) + rho0.i((.., 0, 1))) * &ch2.weights).sum();
        let vxc = ni_obj.make_vxc_pot_with_eff(vxc_eff.view(), SIGMA, 1).unwrap();
        let fxc = ni_obj.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), SIGMA, 1).unwrap();
        let kxc = ni_obj.make_kxc_pot_with_eff(kxc_eff.view(), rho1.view(), rho2.view(), SIGMA, 1).unwrap();
        assert!((exc - -5.2725625947).abs() < 1e-6);
        fp_assert_eq!(vxc.view(), -13.134537099, 1e-6);
        fp_assert_eq!(fxc.view(), -0.13792205114629885, 1e-6);
        fp_assert_eq!(kxc.view(), 0.2770362015666589, 1e-6);
    }

    #[rstest]
    fn test_tau(ch2: &Ch2Molecule, perturbed_dm: &Ch2PerturbedDM) {
        let mut ni_obj = ch2.build_ni_obj();
        let rho0 = ni_obj.make_rho_from_dm(&dm0_list(ch2), TAU).unwrap();
        let dm1_list: Vec<_> = perturbed_dm.dm1_flat.axes_iter(-1).collect();
        let dm2_list: Vec<_> = perturbed_dm.dm2_flat.axes_iter(-1).collect();
        let rho1: Tsr =
            ni_obj.make_rho_from_dm(&dm1_list, TAU).unwrap().into_shape([ch2.ngrids, 5, 2, perturbed_dm.ncomp1]);
        let rho2: Tsr =
            ni_obj.make_rho_from_dm(&dm2_list, TAU).unwrap().into_shape([ch2.ngrids, 5, 2, perturbed_dm.ncomp2]);

        let xc_func = LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSSH", Polarized);
        let xc_eff = libxc_eval_eff(&xc_func, rho0.view(), 3, true).unwrap();
        let [exc_eff, vxc_eff, fxc_eff, kxc_eff] = xc_eff.try_into().unwrap();
        let exc = (exc_eff * (rho0.i((.., 0, 0)) + rho0.i((.., 0, 1))) * &ch2.weights).sum();
        let vxc = ni_obj.make_vxc_pot_with_eff(vxc_eff.view(), TAU, 1).unwrap();
        let fxc = ni_obj.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), TAU, 1).unwrap();
        let kxc = ni_obj.make_kxc_pot_with_eff(kxc_eff.view(), rho1.view(), rho2.view(), TAU, 1).unwrap();
        assert!((exc - -4.9638946892).abs() < 1e-6);
        fp_assert_eq!(vxc.view(), -12.384391087, 1e-6);
        fp_assert_eq!(fxc.view(), 31.692895267010428, 1e-5);
        fp_assert_eq!(kxc.view(), 6528.81912736829, 1e-4);
    }
}

// ---------------------------------------------------------------------------
// TestXCPotPure — naive vs optimized pure function comparison (UKS)
// ---------------------------------------------------------------------------

mod test_xcpot_pure {
    use super::*;
    use rstsr_showcase_dft_grids::numint_matmul::pure_xcpot::{
        uks_fxc_pot_with_output, uks_kxc_pot_with_output, uks_vxc_pot_with_output,
    };
    use rstsr_showcase_dft_grids::numint_matmul::pure_xcpot_naive::{
        uks_fxc_pot_with_output_naive, uks_kxc_pot_with_output_naive, uks_vxc_pot_with_output_naive,
    };

    fn make_out(shape: &[usize], ch2: &Ch2Molecule) -> Tsr {
        let device = ch2.rdm1.device().clone();
        rt::asarray((vec![0.0; shape.iter().product()], shape.to_vec(), &device))
    }

    #[rstest]
    fn test_uks_vxc_pot_naive_vs_optimized(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        for den_type in [RHO, SIGMA, TAU] {
            let dm0_list = [ch2.rdm1.i((.., .., 0)), ch2.rdm1.i((.., .., 1))];
            let rho0 = ni_obj.make_rho_from_dm(&dm0_list, den_type).unwrap();
            let xc_func = LibXCFunctional::from_identifier(
                match den_type {
                    RHO => "LDA_X",
                    SIGMA => "GGA_X_PBE",
                    TAU => "HYB_MGGA_XC_TPSSH",
                    _ => unreachable!(),
                },
                Polarized,
            );
            let xc_eff = libxc_eval_eff(&xc_func, rho0.view(), 1, true).unwrap();
            let ao = ni_obj.prepare_ao(den_type.num_ao_deriv());
            let nao = ao.shape()[1];

            let mut out_naive = make_out(&[nao, nao, 2], ch2);
            let mut out_opt = make_out(&[nao, nao, 2], ch2);
            uks_vxc_pot_with_output_naive(
                den_type,
                xc_eff[1].view(),
                ao.view(),
                ch2.weights.view(),
                out_naive.view_mut(),
            )
            .unwrap();
            uks_vxc_pot_with_output(den_type, xc_eff[1].view(), ao.view(), ch2.weights.view(), out_opt.view_mut())
                .unwrap();
            let diff = (&out_naive - &out_opt).abs().max();
            assert!(diff < 1e-10, "{:?} vxc naive vs opt max diff = {:.3e}", den_type, diff);
        }
    }

    #[rstest]
    fn test_uks_fxc_pot_naive_vs_optimized(ch2: &Ch2Molecule, perturbed_dm: &Ch2PerturbedDM) {
        let mut ni_obj = ch2.build_ni_obj();
        for den_type in [RHO, SIGMA, TAU] {
            let dm0_list = [ch2.rdm1.i((.., .., 0)), ch2.rdm1.i((.., .., 1))];
            let rho0 = ni_obj.make_rho_from_dm(&dm0_list, den_type).unwrap();
            let dm1_list: Vec<_> = perturbed_dm.dm1_flat.axes_iter(-1).collect();
            let rho1 = ni_obj.make_rho_from_dm(&dm1_list, den_type).unwrap().into_shape([
                ch2.ngrids,
                den_type.num_nvar(),
                2,
                perturbed_dm.ncomp1,
            ]);
            let xc_func = LibXCFunctional::from_identifier(
                match den_type {
                    RHO => "LDA_X",
                    SIGMA => "GGA_X_PBE",
                    TAU => "HYB_MGGA_XC_TPSSH",
                    _ => unreachable!(),
                },
                Polarized,
            );
            let xc_eff = libxc_eval_eff(&xc_func, rho0.view(), 2, true).unwrap();
            let ao = ni_obj.prepare_ao(den_type.num_ao_deriv());
            let nao = ao.shape()[1];

            let mut out_naive = make_out(&[nao, nao, 2, perturbed_dm.ncomp1], ch2);
            let mut out_opt = make_out(&[nao, nao, 2, perturbed_dm.ncomp1], ch2);
            uks_fxc_pot_with_output_naive(
                den_type,
                xc_eff[2].view(),
                rho1.view(),
                ao.view(),
                ch2.weights.view(),
                out_naive.view_mut(),
            )
            .unwrap();
            uks_fxc_pot_with_output(
                den_type,
                xc_eff[2].view(),
                rho1.view(),
                ao.view(),
                ch2.weights.view(),
                out_opt.view_mut(),
            )
            .unwrap();
            let diff = (&out_naive - &out_opt).abs().max();
            assert!(diff < 1e-10, "{:?} fxc naive vs opt max diff = {:.3e}", den_type, diff);
        }
    }

    #[rstest]
    fn test_uks_kxc_pot_naive_vs_optimized(ch2: &Ch2Molecule, perturbed_dm: &Ch2PerturbedDM) {
        let mut ni_obj = ch2.build_ni_obj();
        for den_type in [RHO, SIGMA, TAU] {
            let dm0_list = [ch2.rdm1.i((.., .., 0)), ch2.rdm1.i((.., .., 1))];
            let rho0 = ni_obj.make_rho_from_dm(&dm0_list, den_type).unwrap();
            let dm1_list: Vec<_> = perturbed_dm.dm1_flat.axes_iter(-1).collect();
            let dm2_list: Vec<_> = perturbed_dm.dm2_flat.axes_iter(-1).collect();
            let rho1 = ni_obj.make_rho_from_dm(&dm1_list, den_type).unwrap().into_shape([
                ch2.ngrids,
                den_type.num_nvar(),
                2,
                perturbed_dm.ncomp1,
            ]);
            let rho2 = ni_obj.make_rho_from_dm(&dm2_list, den_type).unwrap().into_shape([
                ch2.ngrids,
                den_type.num_nvar(),
                2,
                perturbed_dm.ncomp2,
            ]);
            let xc_func = LibXCFunctional::from_identifier(
                match den_type {
                    RHO => "LDA_X",
                    SIGMA => "GGA_X_PBE",
                    TAU => "HYB_MGGA_XC_TPSSH",
                    _ => unreachable!(),
                },
                Polarized,
            );
            let xc_eff = libxc_eval_eff(&xc_func, rho0.view(), 3, true).unwrap();
            let ao = ni_obj.prepare_ao(den_type.num_ao_deriv());
            let nao = ao.shape()[1];

            let mut out_naive = make_out(&[nao, nao, 2, perturbed_dm.ncomp1, perturbed_dm.ncomp2], ch2);
            let mut out_opt = make_out(&[nao, nao, 2, perturbed_dm.ncomp1, perturbed_dm.ncomp2], ch2);
            uks_kxc_pot_with_output_naive(
                den_type,
                xc_eff[3].view(),
                rho1.view(),
                rho2.view(),
                ao.view(),
                ch2.weights.view(),
                out_naive.view_mut(),
            )
            .unwrap();
            uks_kxc_pot_with_output(
                den_type,
                xc_eff[3].view(),
                rho1.view(),
                rho2.view(),
                ao.view(),
                ch2.weights.view(),
                out_opt.view_mut(),
            )
            .unwrap();
            let diff = (&out_naive - &out_opt).abs().max();
            assert!(diff < 1e-10, "{:?} kxc naive vs opt max diff = {:.3e}", den_type, diff);
        }
    }
}
