mod test_util;

use libcint::prelude::*;
use rstsr_showcase_dft_grids::prelude::*;
use test_util::*;

#[test]
fn get_rho_from_dm_with_output() {
    let mol_token = r#"
        atom = "O; H 1 0.94; H 1 0.94 2 104.5"
        basis = "def2-TZVP"
    "#;
    let coords = read_npz("h2o.npz", "coords").into_reverse_axes();
    let weights = read_npz("h2o.npz", "weights").into_reverse_axes();
    let ngrids = weights.shape()[0];
    let dm = read_npz("h2o.npz", "rdm1").into_reverse_axes();
    let cint = CIntMol::from_toml(mol_token);
    // change coord Vec<f64> to Vec<[f64; 3]>
    let coords_array = coords.to_owned().into_pack_array::<3>(0).into_vec();
    let mut ni_obj = NIMatMul::new(&cint.cint, &coords_array, &weights.to_vec());
    // rho (lda)
    let rho_lda = ni_obj.get_rho_from_dm_with_output(dm.unsqueeze(-1), NIXCType::LDA).unwrap();
    assert_eq!(rho_lda.shape(), &[ngrids, 1, 1]);
    assert!((fp(rho_lda.view()) - -438.0303348067822).abs() < 1e-6);
    // rho (gga)
    let rho_gga = ni_obj.get_rho_from_dm_with_output(dm.unsqueeze(-1), NIXCType::GGA).unwrap();
    assert_eq!(rho_gga.shape(), &[ngrids, 4, 1]);
    assert!((fp(rho_gga.view()) - 25704.14480085445).abs() < 1e-6);
}
