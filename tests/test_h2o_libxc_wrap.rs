mod test_util;

use libcint::prelude::*;
use libxc::prelude::*;
use rstsr::prelude::*;
use rstsr_showcase_dft_grids::prelude::*;
use test_util::*;

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

    let rho_tau = ni_obj.make_rho_from_dm(&[rdm1.view()], NIDenType::TAU).unwrap();
    let xc_func = LibXCFunctional::from_identifier("hyb_mgga_xc_tpssh", LibXCSpin::Unpolarized);
    let (xc_output, xc_layout) = libxc_eval_inner(&xc_func, rho_tau.i((.., .., 0)), 2).unwrap();
    println!("xc_output: {:?}, xc_layout: {:?}", xc_output.len(), xc_layout);
    // first print libxc outputs
    println!("xc_output: {:?}, xc_layout: {:?}", xc_output.len(), xc_layout);
    for out_name in xc_layout.component_names() {
        let r = xc_layout.get(out_name).unwrap();
        let arr = rt::asarray(&xc_output[r]);
        println!("{}:\n{:13.5e}", out_name, arr);
    }

    use rstsr_showcase_dft_grids::xceff::xc_deriv::transform_xc_inner;
    let ngrids = weights.shape()[0];
    let xc_val_comps = xc_output.len() / ngrids;
    let xc_val = rt::asarray((xc_output, [ngrids, xc_val_comps]));
    let xc_eff = transform_xc_inner(rho_tau.view(), xc_val.view(), NIDenType::TAU, LibXCSpin::Unpolarized, 2);
    println!("xc_eff shape: {:?}", xc_eff.shape());
    println!("xc_eff:\n{:13.5e}", xc_eff.reverse_axes());
}
