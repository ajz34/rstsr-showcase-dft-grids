mod test_util;

use libcint::prelude::*;
use libxc::prelude::*;
use rstsr::prelude::*;
use rstsr_showcase_dft_grids::prelude::*;
use test_util::*;

#[test]
fn test_ch2_eval_xc_inner() {
    let mol_token = r#"
        atom = "C; H 1 1.09; H 1 1.09 2 109.5"
        basis = "def2-TZVP"
    "#;
    let coords = read_npz("ch2.npz", "coords").into_reverse_axes();
    let weights = read_npz("ch2.npz", "weights").into_reverse_axes();
    let rdm1 = read_npz("ch2.npz", "rdm1").into_reverse_axes();
    let cint = CIntMol::from_toml(mol_token);
    // change coord Vec<f64> to Vec<[f64; 3]>
    let coords_array = coords.to_owned().into_pack_array::<3>(0).into_vec();
    let mut ni_obj = NIMatMul::new(&cint.cint, &coords_array, &weights.to_vec());
    let ngrids = weights.shape()[0];

    let rho_tau = ni_obj.make_rho_from_dm(&[rdm1.i((.., .., 0)), rdm1.i((.., .., 1))], NIDenType::TAU).unwrap();
    let xc_func = LibXCFunctional::from_identifier("hyb_mgga_xc_tpssh", LibXCSpin::Polarized);
    let (xc_output, xc_layout) = libxc_eval_inner(&xc_func, rho_tau.view(), 2).unwrap();
    // first print libxc outputs
    println!("xc_output: {:?}, xc_layout: {:?}", xc_output.len(), xc_layout);
    for out_name in xc_layout.component_names() {
        let r = xc_layout.get(out_name).unwrap();
        let rlen = r.end - r.start;
        let arr = rt::asarray((&xc_output[r], [rlen / ngrids, ngrids]));
        println!("{}:\n{:13.5e}", out_name, arr);
    }
}
