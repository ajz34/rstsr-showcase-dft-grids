mod test_util;

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
fn test_ch2_xcpot() {
    let mol_token = r#"
        atom = "C; H 1 0.94; H 1 0.94 2 104.5"
        basis = "def2-TZVP"
    "#;
    let coords = read_npz("ch2.npz", "coords").into_reverse_axes();
    let weights = read_npz("ch2.npz", "weights").into_reverse_axes();
    let rdm1 = read_npz("ch2.npz", "rdm1").into_reverse_axes();
    let cint = CIntMol::from_toml(mol_token);
    let coords_array = coords.to_owned().into_pack_array::<3>(0).into_vec();
    let mut ni_obj = NIMatMul::new(&cint.cint, &coords_array, &weights.to_vec());

    let device = rdm1.device().clone();
    let get_intor = |name: &str| {
        let (out, shape) = ni_obj.cint.integrate(name, None, None).into();
        rt::asarray((out, shape, &device))
    };

    // Construct perturbed DMs for UKS
    // dm1_broadcast: [nao, nao, 2_spin, 3_comp], dm2_broadcast: [nao, nao, 2_spin, 9_comp]
    let int1e_r_giao: Tsr = get_intor("int1e_r") + get_intor("int1e_giao_irjxp");
    let int1e_rr: Tsr = get_intor("int1e_rr");
    let ncomp1 = int1e_r_giao.shape()[int1e_r_giao.ndim() - 1]; // 3
    let ncomp2 = int1e_rr.shape()[int1e_rr.ndim() - 1]; // 9
    let nao = rdm1.shape()[0];

    let dm1_raw = int1e_r_giao.i((.., .., None, ..)) * rdm1.i((.., .., .., None)); // [nao, nao, 1, 3] * [nao, nao, 2, 1] → [nao, nao, 2, 3]
    let dm1: Tsr = 0.5 * (&dm1_raw + dm1_raw.swapaxes(0, 1));
    let dm2_raw = int1e_rr.i((.., .., None, ..)) * rdm1.i((.., .., .., None)); // [nao, nao, 1, 9] * [nao, nao, 2, 1] → [nao, nao, 2, 9]
    let dm2: Tsr = 0.5 * (&dm2_raw + dm2_raw.swapaxes(0, 1));

    let dm0_list = [rdm1.i((.., .., 0)), rdm1.i((.., .., 1))];
    let ngrids = weights.shape()[0];
    // Flatten spin+comp axes for make_rho_from_dm: column-major merge gives
    // (s=0,c=0),(s=1,c=0),(s=0,c=1),...
    let dm1_flat: Tsr = dm1.into_shape([nao, nao, 2 * ncomp1]);
    let dm2_flat: Tsr = dm2.into_shape([nao, nao, 2 * ncomp2]);
    let dm1_list = dm1_flat.axes_iter(-1).collect::<Vec<_>>();
    let dm2_list = dm2_flat.axes_iter(-1).collect::<Vec<_>>();

    // --- rho (lda, spin=1) ---
    let rho0 = ni_obj.make_rho_from_dm(&dm0_list, RHO).unwrap();
    let rho1: Tsr = ni_obj.make_rho_from_dm(&dm1_list, RHO).unwrap().into_shape([ngrids, 1, 2, ncomp1]);
    let rho2: Tsr = ni_obj.make_rho_from_dm(&dm2_list, RHO).unwrap().into_shape([ngrids, 1, 2, ncomp2]);

    let xc_func = LibXCFunctional::from_identifier("LDA_X", Polarized);
    let xc_eff = libxc_eval_eff_serial(&xc_func, rho0.view(), 3).unwrap();
    let [exc_eff, vxc_eff, fxc_eff, kxc_eff] = xc_eff.try_into().unwrap();
    let exc = (exc_eff * (rho0.i((.., 0, 0)) + rho0.i((.., 0, 1))) * &weights).sum();
    let vxc = ni_obj.make_vxc_pot_with_eff(vxc_eff.view(), RHO, 1).unwrap();
    let fxc = ni_obj.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), RHO, 1).unwrap();
    let kxc = ni_obj.make_kxc_pot_with_eff(kxc_eff.view(), rho1.view(), rho2.view(), RHO, 1).unwrap();
    assert!((exc - -4.7040426008).abs() < 1e-6);
    assert!((fp(vxc.view()) - -12.7427734694).abs() < 1e-6);
    assert!((fp(fxc.view()) - -0.2560478462754152).abs() < 1e-6);
    assert!((fp(kxc.view()) - 0.40388776601225995).abs() < 1e-6);

    // --- sigma (gga, spin=1) ---
    let rho0 = ni_obj.make_rho_from_dm(&dm0_list, SIGMA).unwrap();
    let rho1: Tsr = ni_obj.make_rho_from_dm(&dm1_list, SIGMA).unwrap().into_shape([ngrids, 4, 2, ncomp1]);
    let rho2: Tsr = ni_obj.make_rho_from_dm(&dm2_list, SIGMA).unwrap().into_shape([ngrids, 4, 2, ncomp2]);

    let xc_func = LibXCFunctional::from_identifier("GGA_X_PBE", Polarized);
    let xc_eff = libxc_eval_eff_serial(&xc_func, rho0.view(), 3).unwrap();
    let [exc_eff, vxc_eff, fxc_eff, kxc_eff] = xc_eff.try_into().unwrap();
    let exc = (exc_eff * (rho0.i((.., 0, 0)) + rho0.i((.., 0, 1))) * &weights).sum();
    let vxc = ni_obj.make_vxc_pot_with_eff(vxc_eff.view(), SIGMA, 1).unwrap();
    let fxc = ni_obj.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), SIGMA, 1).unwrap();
    let kxc = ni_obj.make_kxc_pot_with_eff(kxc_eff.view(), rho1.view(), rho2.view(), SIGMA, 1).unwrap();
    assert!((exc - -5.2725625947).abs() < 1e-6);
    assert!((fp(vxc.view()) - -13.134537099).abs() < 1e-6);
    assert!((fp(fxc.view()) - -0.13792205114629885).abs() < 1e-6);
    assert!((fp(kxc.view()) - 0.2770362015666589).abs() < 1e-6);

    // --- tau (mgga, spin=1) ---
    let rho0 = ni_obj.make_rho_from_dm(&dm0_list, TAU).unwrap();
    let rho1: Tsr = ni_obj.make_rho_from_dm(&dm1_list, TAU).unwrap().into_shape([ngrids, 5, 2, ncomp1]);
    let rho2: Tsr = ni_obj.make_rho_from_dm(&dm2_list, TAU).unwrap().into_shape([ngrids, 5, 2, ncomp2]);

    let xc_func = LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSSH", Polarized);
    let xc_eff = libxc_eval_eff_serial(&xc_func, rho0.view(), 3).unwrap();
    let [exc_eff, vxc_eff, fxc_eff, kxc_eff] = xc_eff.try_into().unwrap();
    let exc = (exc_eff * (rho0.i((.., 0, 0)) + rho0.i((.., 0, 1))) * &weights).sum();
    let vxc = ni_obj.make_vxc_pot_with_eff(vxc_eff.view(), TAU, 1).unwrap();
    let fxc = ni_obj.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), TAU, 1).unwrap();
    let kxc = ni_obj.make_kxc_pot_with_eff(kxc_eff.view(), rho1.view(), rho2.view(), TAU, 1).unwrap();
    assert!((exc - -4.9638946892).abs() < 1e-6);
    assert!((fp(vxc.view()) - -12.384391087).abs() < 1e-6);
    assert!((fp(fxc.view()) - 31.692895267010428).abs() < 1e-5);
    assert!((fp(kxc.view()) - 6528.81912736829).abs() < 1e-4);
}
