mod test_util;

use itertools::Itertools;
use libxc::prelude::*;
use rstest::*;
use rstsr::prelude::*;
use rstsr_showcase_dft_grids::prelude::*;
use test_util::*;

use LibXCSpin::*;
use NIDenType::*;

type Tsr<T = f64> = Tensor<T, DeviceFaer, IxD>;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

pub struct H2OPerturbedDM {
    pub dm1_flat: Tsr,
    pub dm2_flat: Tsr,
    pub ncomp1: usize,
    pub ncomp2: usize,
}

#[fixture]
#[once]
fn h2o() -> H2OMolecule {
    H2OMolecule::load()
}

#[fixture]
#[once]
fn perturbed_dm(h2o: &H2OMolecule) -> H2OPerturbedDM {
    let device = h2o.rdm1.device().clone();
    let get_intor = |name: &str| {
        let (out, shape) = h2o.cint().integrate(name, None, None).into();
        rt::asarray((out, shape, &device))
    };
    let dm1: Tsr = (get_intor("int1e_r") + get_intor("int1e_giao_irjxp")) * &h2o.rdm1;
    let dm2: Tsr = get_intor("int1e_rr") * &h2o.rdm1;
    let dm1: Tsr = 0.5 * (&dm1 + dm1.swapaxes(0, 1));
    let dm2: Tsr = 0.5 * (&dm2 + dm2.swapaxes(0, 1));
    let ncomp1 = dm1.shape()[dm1.ndim() - 1];
    let ncomp2 = dm2.shape()[dm2.ndim() - 1];

    H2OPerturbedDM { dm1_flat: dm1, dm2_flat: dm2, ncomp1, ncomp2 }
}

// ---------------------------------------------------------------------------
// TestXCPot
// ---------------------------------------------------------------------------

mod test_xcpot {
    use super::*;

    fn dm0_view(h2o: &H2OMolecule) -> TsrView<'_> {
        h2o.rdm1.view()
    }

    #[rstest]
    fn test_rho(h2o: &H2OMolecule, perturbed_dm: &H2OPerturbedDM) {
        let mut ni_obj = h2o.build_ni_obj();
        let rho0 = ni_obj.make_rho_from_dm(&[dm0_view(h2o)], RHO).unwrap().into_squeeze(-1);
        let dm1_list = perturbed_dm.dm1_flat.axes_iter(-1).collect_vec();
        let dm2_list = perturbed_dm.dm2_flat.axes_iter(-1).collect_vec();
        let rho1 = ni_obj.make_rho_from_dm(&dm1_list, RHO).unwrap();
        let rho2 = ni_obj.make_rho_from_dm(&dm2_list, RHO).unwrap();

        let xc_func = LibXCFunctional::from_identifier("LDA_X", Unpolarized);
        let xc_eff = libxc_eval_eff(&xc_func, rho0.view(), 3, true).unwrap();
        let [exc_eff, vxc_eff, fxc_eff, kxc_eff] = xc_eff.try_into().unwrap();
        let exc = (exc_eff * rho0.i((.., 0)) * &h2o.weights).sum();
        let vxc = ni_obj.make_vxc_pot_with_eff(vxc_eff.view(), RHO, 0).unwrap();
        let fxc = ni_obj.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), RHO, 0).unwrap();
        let kxc = ni_obj.make_kxc_pot_with_eff(kxc_eff.view(), rho1.view(), rho2.view(), RHO, 0).unwrap();
        assert!((exc - -8.1384975323).abs() < 1e-6);
        fp_assert_eq!(vxc.view(),-27.2331156537, 1e-6);
        fp_assert_eq!(fxc.view(),-0.09693300035135462, 1e-6);
        fp_assert_eq!(kxc.view(),0.3789165091826895, 1e-6);
    }

    #[rstest]
    fn test_sigma(h2o: &H2OMolecule, perturbed_dm: &H2OPerturbedDM) {
        let mut ni_obj = h2o.build_ni_obj();
        let rho0 = ni_obj.make_rho_from_dm(&[dm0_view(h2o)], SIGMA).unwrap().into_squeeze(-1);
        let dm1_list = perturbed_dm.dm1_flat.axes_iter(-1).collect_vec();
        let dm2_list = perturbed_dm.dm2_flat.axes_iter(-1).collect_vec();
        let rho1 = ni_obj.make_rho_from_dm(&dm1_list, SIGMA).unwrap();
        let rho2 = ni_obj.make_rho_from_dm(&dm2_list, SIGMA).unwrap();

        let xc_func = LibXCFunctional::from_identifier("GGA_X_PBE", Unpolarized);
        let xc_eff = libxc_eval_eff(&xc_func, rho0.view(), 3, true).unwrap();
        let [exc_eff, vxc_eff, fxc_eff, kxc_eff] = xc_eff.try_into().unwrap();
        let exc = (exc_eff * rho0.i((.., 0)) * &h2o.weights).sum();
        let vxc = ni_obj.make_vxc_pot_with_eff(vxc_eff.view(), SIGMA, 0).unwrap();
        let fxc = ni_obj.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), SIGMA, 0).unwrap();
        let kxc = ni_obj.make_kxc_pot_with_eff(kxc_eff.view(), rho1.view(), rho2.view(), SIGMA, 0).unwrap();
        assert!((exc - -8.9542650216).abs() < 1e-6);
        fp_assert_eq!(vxc.view(),-28.6270372279, 1e-6);
        fp_assert_eq!(fxc.view(),-0.10389233031803395, 1e-6);
        fp_assert_eq!(kxc.view(),0.40594124509389706, 1e-6);
    }

    #[rstest]
    fn test_tau(h2o: &H2OMolecule, perturbed_dm: &H2OPerturbedDM) {
        let mut ni_obj = h2o.build_ni_obj();
        let rho0 = ni_obj.make_rho_from_dm(&[dm0_view(h2o)], TAU).unwrap().into_squeeze(-1);
        let dm1_list = perturbed_dm.dm1_flat.axes_iter(-1).collect_vec();
        let dm2_list = perturbed_dm.dm2_flat.axes_iter(-1).collect_vec();
        let rho1 = ni_obj.make_rho_from_dm(&dm1_list, TAU).unwrap();
        let rho2 = ni_obj.make_rho_from_dm(&dm2_list, TAU).unwrap();

        let xc_func = LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSSH", Unpolarized);
        let xc_eff = libxc_eval_eff(&xc_func, rho0.view(), 3, true).unwrap();
        let [exc_eff, vxc_eff, fxc_eff, kxc_eff] = xc_eff.try_into().unwrap();
        let exc = (exc_eff * rho0.i((.., 0)) * &h2o.weights).sum();
        let vxc = ni_obj.make_vxc_pot_with_eff(vxc_eff.view(), TAU, 0).unwrap();
        let fxc = ni_obj.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), TAU, 0).unwrap();
        let kxc = ni_obj.make_kxc_pot_with_eff(kxc_eff.view(), rho1.view(), rho2.view(), TAU, 0).unwrap();
        assert!((exc - -8.4667246286).abs() < 1e-6);
        fp_assert_eq!(vxc.view(),-26.3517912584, 1e-6);
        fp_assert_eq!(fxc.view(),-0.09110536214579629, 1e-6);
        fp_assert_eq!(kxc.view(),0.5466595210064285, 1e-6);
    }
}

// ---------------------------------------------------------------------------
// TestXCPotPure — naive vs optimized pure function comparison
// ---------------------------------------------------------------------------

mod test_xcpot_pure {
    use super::*;
    use rstsr_showcase_dft_grids::numint_matmul::pure_xcpot::{
        rks_fxc_pot_with_output, rks_kxc_pot_with_output, rks_vxc_pot_with_output,
    };
    use rstsr_showcase_dft_grids::numint_matmul::pure_xcpot_naive::{
        rks_fxc_pot_with_output_naive, rks_kxc_pot_with_output_naive, rks_vxc_pot_with_output_naive,
    };

    fn make_out(shape: &[usize], h2o: &H2OMolecule) -> Tsr {
        let device = h2o.rdm1.device().clone();
        rt::asarray((vec![0.0; shape.iter().product()], shape.to_vec(), &device))
    }

    #[rstest]
    fn test_rks_vxc_pot_naive_vs_optimized(
        h2o: &H2OMolecule,
    ) {
        let mut ni_obj = h2o.build_ni_obj();
        for den_type in [RHO, SIGMA, TAU] {
            let rho0 = ni_obj.make_rho_from_dm(&[h2o.rdm1.view()], den_type).unwrap().into_squeeze(-1);
            let xc_func = LibXCFunctional::from_identifier(
                match den_type { RHO => "LDA_X", SIGMA => "GGA_X_PBE", TAU => "HYB_MGGA_XC_TPSSH", _ => unreachable!() },
                Unpolarized,
            );
            let xc_eff = libxc_eval_eff(&xc_func, rho0.view(), 1, true).unwrap();
            let ao = ni_obj.prepare_ao(den_type.num_ao_deriv());
            let nao = ao.shape()[1];

            let mut out_naive = make_out(&[nao, nao], h2o);
            let mut out_opt = make_out(&[nao, nao], h2o);
            rks_vxc_pot_with_output_naive(
                den_type, xc_eff[1].view(), ao.view(), h2o.weights.view(), out_naive.view_mut(),
            ).unwrap();
            rks_vxc_pot_with_output(
                den_type, xc_eff[1].view(), ao.view(), h2o.weights.view(), out_opt.view_mut(),
            ).unwrap();
            let diff = (&out_naive - &out_opt).abs().max();
            assert!(diff < 1e-10, "{:?} vxc naive vs opt max diff = {:.3e}", den_type, diff);
        }
    }

    #[rstest]
    fn test_rks_fxc_pot_naive_vs_optimized(
        h2o: &H2OMolecule,
        perturbed_dm: &H2OPerturbedDM,
    ) {
        let mut ni_obj = h2o.build_ni_obj();
        for den_type in [RHO, SIGMA, TAU] {
            let rho0 = ni_obj.make_rho_from_dm(&[h2o.rdm1.view()], den_type).unwrap().into_squeeze(-1);
            let dm1_list: Vec<_> = perturbed_dm.dm1_flat.axes_iter(-1).collect();
            let rho1 = ni_obj.make_rho_from_dm(&dm1_list, den_type).unwrap();
            let xc_func = LibXCFunctional::from_identifier(
                match den_type { RHO => "LDA_X", SIGMA => "GGA_X_PBE", TAU => "HYB_MGGA_XC_TPSSH", _ => unreachable!() },
                Unpolarized,
            );
            let xc_eff = libxc_eval_eff(&xc_func, rho0.view(), 2, true).unwrap();
            let ao = ni_obj.prepare_ao(den_type.num_ao_deriv());
            let nao = ao.shape()[1];

            let mut out_naive = make_out(&[nao, nao, perturbed_dm.ncomp1], h2o);
            let mut out_opt = make_out(&[nao, nao, perturbed_dm.ncomp1], h2o);
            rks_fxc_pot_with_output_naive(
                den_type, xc_eff[2].view(), rho1.view(), ao.view(), h2o.weights.view(), out_naive.view_mut(),
            ).unwrap();
            rks_fxc_pot_with_output(
                den_type, xc_eff[2].view(), rho1.view(), ao.view(), h2o.weights.view(), out_opt.view_mut(),
            ).unwrap();
            let diff = (&out_naive - &out_opt).abs().max();
            assert!(diff < 1e-10, "{:?} fxc naive vs opt max diff = {:.3e}", den_type, diff);
        }
    }

    #[rstest]
    fn test_rks_kxc_pot_naive_vs_optimized(
        h2o: &H2OMolecule,
        perturbed_dm: &H2OPerturbedDM,
    ) {
        let mut ni_obj = h2o.build_ni_obj();
        for den_type in [RHO, SIGMA, TAU] {
            let rho0 = ni_obj.make_rho_from_dm(&[h2o.rdm1.view()], den_type).unwrap().into_squeeze(-1);
            let dm1_list: Vec<_> = perturbed_dm.dm1_flat.axes_iter(-1).collect();
            let dm2_list: Vec<_> = perturbed_dm.dm2_flat.axes_iter(-1).collect();
            let rho1 = ni_obj.make_rho_from_dm(&dm1_list, den_type).unwrap();
            let rho2 = ni_obj.make_rho_from_dm(&dm2_list, den_type).unwrap();
            let xc_func = LibXCFunctional::from_identifier(
                match den_type { RHO => "LDA_X", SIGMA => "GGA_X_PBE", TAU => "HYB_MGGA_XC_TPSSH", _ => unreachable!() },
                Unpolarized,
            );
            let xc_eff = libxc_eval_eff(&xc_func, rho0.view(), 3, true).unwrap();
            let ao = ni_obj.prepare_ao(den_type.num_ao_deriv());
            let nao = ao.shape()[1];

            let mut out_naive = make_out(&[nao, nao, perturbed_dm.ncomp1, perturbed_dm.ncomp2], h2o);
            let mut out_opt = make_out(&[nao, nao, perturbed_dm.ncomp1, perturbed_dm.ncomp2], h2o);
            rks_kxc_pot_with_output_naive(
                den_type, xc_eff[3].view(), rho1.view(), rho2.view(), ao.view(), h2o.weights.view(), out_naive.view_mut(),
            ).unwrap();
            rks_kxc_pot_with_output(
                den_type, xc_eff[3].view(), rho1.view(), rho2.view(), ao.view(), h2o.weights.view(), out_opt.view_mut(),
            ).unwrap();
            let diff = (&out_naive - &out_opt).abs().max();
            assert!(diff < 1e-10, "{:?} kxc naive vs opt max diff = {:.3e}", den_type, diff);
        }
    }
}
