mod test_util;

use libcint::prelude::*;
use rstsr_showcase_dft_grids::prelude::*;
use test_util::*;

#[test]
fn get_rho_from_dm() {
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
    // rho
    let out_rho = ni_obj.make_rho_from_dm(dm.unsqueeze(-1), NIDenType::RHO).unwrap();
    assert_eq!(out_rho.shape(), &[ngrids, 1, 1]);
    assert!((fp(out_rho.view()) - -438.0303348067822).abs() < 1e-6);
    // sigma
    let out_sigma = ni_obj.make_rho_from_dm(dm.unsqueeze(-1), NIDenType::SIGMA).unwrap();
    assert_eq!(out_sigma.shape(), &[ngrids, 4, 1]);
    assert!((fp(out_sigma.view()) - 25704.14480085445).abs() < 1e-6);
    // tau
    let out_tau = ni_obj.make_rho_from_dm(dm.unsqueeze(-1), NIDenType::TAU).unwrap();
    assert_eq!(out_tau.shape(), &[ngrids, 5, 1]);
    assert!((fp(out_tau.view()) - 17140.300791589965).abs() < 1e-6);
    // lapl
    let out_lapl = ni_obj.make_rho_from_dm(dm.unsqueeze(-1), NIDenType::LAPL).unwrap();
    assert_eq!(out_lapl.shape(), &[ngrids, 6, 1]);
    assert!((fp(out_lapl.i((.., ..5, ..))) - 17140.300791589965).abs() < 1e-6);
    assert!((fp(out_lapl.i((.., 5, ..))) - 2470300.1875723703).abs() < 1e-4);
}

#[test]
fn get_rho_from_homogeneous_braket() {
    use rstsr::prelude::*;

    let mol_token = r#"
        atom = "O; H 1 0.94; H 1 0.94 2 104.5"
        basis = "def2-TZVP"
    "#;
    let coords = read_npz("h2o.npz", "coords").into_reverse_axes();
    let weights = read_npz("h2o.npz", "weights").into_reverse_axes();
    let ngrids = weights.shape()[0];
    let mo_coeff = read_npz("h2o.npz", "mo_coeff").into_contig(ColMajor);
    let mo_occ = read_npz("h2o.npz", "mo_occ").into_reverse_axes();
    let cint = CIntMol::from_toml(mol_token);
    // change coord Vec<f64> to Vec<[f64; 3]>
    let coords_array = coords.to_owned().into_pack_array::<3>(0).into_vec();
    let mut ni_obj = NIMatMul::new(&cint.cint, &coords_array, &weights.to_vec());
    // generate homogeneous braket (nao, nocc, nset)
    let occ_mask = mo_occ.greater(0.0).into_vec();
    let occ = mo_occ.bool_select(0, &occ_mask);
    let bra = mo_coeff.bool_select(1, &occ_mask) * occ.sqrt().i((None, ..));

    // rho
    let out_rho = ni_obj.make_rho_from_homogeneous_braket(&[bra.view()], NIDenType::RHO).unwrap();
    assert_eq!(out_rho.shape(), &[ngrids, 1, 1]);
    assert!((fp(out_rho.view()) - -438.0303348067822).abs() < 1e-6);
    // sigma
    let out_sigma = ni_obj.make_rho_from_homogeneous_braket(&[bra.view()], NIDenType::SIGMA).unwrap();
    assert_eq!(out_sigma.shape(), &[ngrids, 4, 1]);
    assert!((fp(out_sigma.view()) - 25704.14480085445).abs() < 1e-6);
    // tau
    let out_tau = ni_obj.make_rho_from_homogeneous_braket(&[bra.view()], NIDenType::TAU).unwrap();
    assert_eq!(out_tau.shape(), &[ngrids, 5, 1]);
    assert!((fp(out_tau.view()) - 17140.300791589965).abs() < 1e-6);
    // lapl
    let out_lapl = ni_obj.make_rho_from_homogeneous_braket(&[bra.view()], NIDenType::LAPL).unwrap();
    assert_eq!(out_lapl.shape(), &[ngrids, 6, 1]);
    assert!((fp(out_lapl.i((.., ..5, ..))) - 17140.300791589965).abs() < 1e-6);
    assert!((fp(out_lapl.i((.., 5, ..))) - 2470300.1875723703).abs() < 1e-4);
}
