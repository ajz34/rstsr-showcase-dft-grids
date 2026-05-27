mod test_util;

use itertools::Itertools;
use libcint::prelude::*;
use libxc::prelude::*;
use rstsr::prelude::*;
use rstsr_showcase_dft_grids::prelude::*;
use test_util::*;

use LibXCSpin::*;
use NIDenType::*;

type DeviceTsr = DeviceFaer;
type Tsr<T = f64> = Tensor<T, DeviceTsr, IxD>;

#[test]
fn test_h2o_xcpot() {
    let mol_token = r#"
        atom = "O; H 1 0.94; H 1 0.94 2 104.5"
        basis = "def2-TZVP"
    "#;
    let coords = read_npz("h2o.npz", "coords").into_reverse_axes();
    let weights = read_npz("h2o.npz", "weights").into_reverse_axes();
    let rdm1 = read_npz("h2o.npz", "rdm1").into_reverse_axes();
    let cint = CIntMol::from_toml(mol_token);
    // change coord Vec<f64> to Vec<[f64; 3]>
    let coords_array = coords.to_owned().into_pack_array::<3>(0).into_vec();
    let mut ni_obj = NIMatMul::new(&cint.cint, &coords_array, &weights.to_vec());
    // make fake densities
    let device = rdm1.device().clone();
    let dm0 = rdm1.view();
    let get_intor = |name: &str| {
        let (out, shape) = ni_obj.cint.integrate(name, None, None).into();
        rt::asarray((out, shape, &device))
    };
    let dm1 = (get_intor("int1e_r") + get_intor("int1e_giao_irjxp")) * &dm0;
    let dm2 = get_intor("int1e_rr") * &dm0;
    let dm1: Tsr = 0.5 * (&dm1 + dm1.swapaxes(0, 1));
    let dm2: Tsr = 0.5 * (&dm2 + dm2.swapaxes(0, 1));
    let dm1_list = dm1.axes_iter(-1).collect_vec();
    let dm2_list = dm2.axes_iter(-1).collect_vec();

    // --- rho (lda) ---
    let rho0 = ni_obj.make_rho_from_dm(&[dm0.view()], RHO).unwrap().into_squeeze(-1);
    let rho1 = ni_obj.make_rho_from_dm(&dm1_list, RHO).unwrap();
    let rho2 = ni_obj.make_rho_from_dm(&dm2_list, RHO).unwrap();

    let xc_func = LibXCFunctional::from_identifier("LDA_X", Unpolarized);
    let xc_eff = libxc_eval_eff_serial(&xc_func, rho0.view(), 3).unwrap();
    let [exc_eff, vxc_eff, fxc_eff, kxc_eff] = xc_eff.try_into().unwrap();
    let exc = (exc_eff * rho0.i((.., 0)) * &weights).sum();
    let vxc = ni_obj.make_vxc_pot_with_eff(vxc_eff.view(), RHO, 0).unwrap();
    let fxc = ni_obj.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), RHO, 0).unwrap();
    let kxc = ni_obj.make_kxc_pot_with_eff(kxc_eff.view(), rho1.view(), rho2.view(), RHO, 0).unwrap();
    assert!((exc - -8.1384975323).abs() < 1e-6);
    assert!((fp(vxc.view()) - -27.2331156537).abs() < 1e-6);
    assert!((fp(fxc.view()) - -0.09693300035135462).abs() < 1e-6);
    assert!((fp(kxc.view()) - 0.3789165091826895).abs() < 1e-6);

    // --- sigma (gga) ---
    let time = std::time::Instant::now();
    let rho0 = ni_obj.make_rho_from_dm(&[dm0.view()], SIGMA).unwrap().into_squeeze(-1);
    let rho1 = ni_obj.make_rho_from_dm(&dm1_list, SIGMA).unwrap();
    let rho2 = ni_obj.make_rho_from_dm(&dm2_list, SIGMA).unwrap();
    println!("SIGMA make_rho_from_dm time: {:?}", time.elapsed());

    let time = std::time::Instant::now();
    let xc_func = LibXCFunctional::from_identifier("GGA_X_PBE", Unpolarized);
    let xc_eff = libxc_eval_eff_serial(&xc_func, rho0.view(), 3).unwrap();
    let [exc_eff, vxc_eff, fxc_eff, kxc_eff] = xc_eff.try_into().unwrap();
    println!("SIGMA libxc_eval_eff_serial time: {:?}", time.elapsed());

    let time = std::time::Instant::now();
    let exc = (exc_eff * rho0.i((.., 0)) * &weights).sum();
    let vxc = ni_obj.make_vxc_pot_with_eff(vxc_eff.view(), SIGMA, 0).unwrap();
    let fxc = ni_obj.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), SIGMA, 0).unwrap();
    let kxc = ni_obj.make_kxc_pot_with_eff(kxc_eff.view(), rho1.view(), rho2.view(), SIGMA, 0).unwrap();
    println!("SIGMA make_?xc_pot_with_eff time: {:?}", time.elapsed());

    assert!((exc - -8.9542650216).abs() < 1e-6);
    assert!((fp(vxc.view()) - -28.6270372279).abs() < 1e-6);
    assert!((fp(fxc.view()) - -0.10389233031803395).abs() < 1e-6);
    assert!((fp(kxc.view()) - 0.40594124509389706).abs() < 1e-6);

    // --- tau (mgga) ---
    let time = std::time::Instant::now();
    let rho0 = ni_obj.make_rho_from_dm(&[dm0.view()], TAU).unwrap().into_squeeze(-1);
    let rho1 = ni_obj.make_rho_from_dm(&dm1_list, TAU).unwrap();
    let rho2 = ni_obj.make_rho_from_dm(&dm2_list, TAU).unwrap();
    println!("TAU make_rho_from_dm time: {:?}", time.elapsed());

    let time = std::time::Instant::now();
    let xc_func = LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSSH", Unpolarized);
    let xc_eff = libxc_eval_eff_serial(&xc_func, rho0.view(), 3).unwrap();
    let [exc_eff, vxc_eff, fxc_eff, kxc_eff] = xc_eff.try_into().unwrap();
    println!("TAU libxc_eval_eff_serial time: {:?}", time.elapsed());

    let time = std::time::Instant::now();
    let exc = (exc_eff * rho0.i((.., 0)) * &weights).sum();
    let vxc = ni_obj.make_vxc_pot_with_eff(vxc_eff.view(), TAU, 0).unwrap();
    let fxc = ni_obj.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), TAU, 0).unwrap();
    let kxc = ni_obj.make_kxc_pot_with_eff(kxc_eff.view(), rho1.view(), rho2.view(), TAU, 0).unwrap();
    println!("TAU make_?xc_pot_with_eff time: {:?}", time.elapsed());

    assert!((exc - -8.4667246286).abs() < 1e-6);
    assert!((fp(vxc.view()) - -26.3517912584).abs() < 1e-6);
    assert!((fp(fxc.view()) - -0.09110536214579629).abs() < 1e-6);
    assert!((fp(kxc.view()) - 0.5466595210064285).abs() < 1e-6);
}
