mod test_util;

use libcint::prelude::*;
use libxc::prelude::*;
use rstsr_showcase_dft_grids::prelude::*;
use test_util::*;

use LibXCSpin::*;
use NIDenType::*;

#[test]
fn test_h2o_eval_xc_inner() {
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

    // this density can be utilized by rho/sigma/tau.
    let rho_tau = ni_obj.make_rho_from_dm(&[rdm1.view()], TAU).unwrap().into_squeeze(-1);

    // rho (lda)
    let xc_func = LibXCFunctional::from_identifier("lda_x", Unpolarized);
    let rho_rho = rho_tau.i((.., ..1));
    let xc_eff = libxc_eval_eff(&xc_func, rho_rho.view(), 3).unwrap();
    let fps = [
        fp((&xc_eff[0] * &weights).view()),
        fp((&xc_eff[1] * &weights).view()),
        fp((&xc_eff[2] * &weights * &rho_rho).view()),
        fp((&xc_eff[3] * &weights * &rho_rho * &rho_rho.i((.., None, ..))).view()),
    ];
    println!("LDA xc_eff fps: {:?}", fps);
    assert!((fps[0] - -0.0653646142).abs() < 1e-6);
    assert!((fps[1] - -0.0871528189).abs() < 1e-6);
    assert!((fps[2] - -0.0290509396).abs() < 1e-6);
    assert!((fps[3] - 0.0193672931).abs() < 1e-6);

    // sigma (gga)
    let xc_func = LibXCFunctional::from_identifier("gga_x_pbe", Unpolarized);
    let rho_sigma = rho_tau.i((.., ..4));
    let xc_eff = libxc_eval_eff(&xc_func, rho_sigma.view(), 3).unwrap();
    let fps = [
        fp((&xc_eff[0] * &weights).view()),
        fp((&xc_eff[1] * &weights).view()),
        fp((&xc_eff[2] * &weights * &rho_sigma).view()),
        fp((&xc_eff[3] * &weights * &rho_sigma * &rho_sigma.i((.., None, ..))).view()),
    ];
    println!("GGA xc_eff fps: {:?}", fps);
    assert!((fps[0] - -0.1652985961).abs() < 1e-6);
    assert!((fps[1] - -0.2296325511).abs() < 1e-6);
    assert!((fps[2] - -0.1386410873).abs() < 1e-6);
    assert!((fps[3] - -0.523551635).abs() < 1e-6);

    // tau (meta-GGA tau)
    let xc_func = LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSSH", Unpolarized);
    let xc_eff = libxc_eval_eff(&xc_func, rho_tau.view(), 3).unwrap();
    let fps = [
        fp((&xc_eff[0] * &weights).view()),
        fp((&xc_eff[1] * &weights).view()),
        fp((&xc_eff[2] * &weights * &rho_tau).view()),
        fp((&xc_eff[3] * &weights * &rho_tau * &rho_tau.i((.., None, ..))).view()),
    ];
    println!("meta-GGA tau xc_eff fps: {:?}", fps);
    assert!((fps[0] - -0.1378400498).abs() < 1e-6);
    assert!((fps[1] - -0.1893572125).abs() < 1e-6);
    assert!((fps[2] - -0.1183454954).abs() < 1e-6);
    assert!((fps[3] - -1.0447740367).abs() < 1e-6);
}
