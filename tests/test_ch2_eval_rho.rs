mod test_util;

use libcint::prelude::*;
use rstsr_showcase_dft_grids::prelude::*;
use test_util::*;

#[test]
fn get_rho_from_dm() {
    let mol_token = r#"
        atom = "C; H 1 0.94; H 1 0.94 2 104.5"
        basis = "def2-TZVP"
    "#;
    let coords = read_npz("ch2.npz", "coords").into_reverse_axes();
    let weights = read_npz("ch2.npz", "weights").into_reverse_axes();
    let ngrids = weights.shape()[0];
    let dm = read_npz("ch2.npz", "rdm1").into_reverse_axes();
    let cint = CIntMol::from_toml(mol_token);
    // change coord Vec<f64> to Vec<[f64; 3]>
    let coords_array = coords.to_owned().into_pack_array::<3>(0).into_vec();
    let mut ni_obj = NIMatMul::new(&cint.cint, &coords_array, &weights.to_vec());
    let dm_list = [dm.i((.., .., 0)), dm.i((.., .., 1))];
    // rho
    let out_rho = ni_obj.make_rho_from_dm(&dm_list, NIDenType::RHO).unwrap();
    assert_eq!(out_rho.shape(), &[ngrids, 1, 2]);
    assert!((fp(out_rho.view()) - 90.1600267407401).abs() < 1e-6);
    // sigma
    let out_sigma = ni_obj.make_rho_from_dm(&dm_list, NIDenType::SIGMA).unwrap();
    assert_eq!(out_sigma.shape(), &[ngrids, 4, 2]);
    assert!((fp(out_sigma.view()) - 369.8178428546338).abs() < 1e-6);
    // tau
    let out_tau = ni_obj.make_rho_from_dm(&dm_list, NIDenType::TAU).unwrap();
    assert_eq!(out_tau.shape(), &[ngrids, 5, 2]);
    assert!((fp(out_tau.view()) - 5587.859016487346).abs() < 1e-6);
    // lapl
    let out_lapl = ni_obj.make_rho_from_dm(&dm_list, NIDenType::LAPL).unwrap();
    assert_eq!(out_lapl.shape(), &[ngrids, 6, 2]);
    assert!((fp(out_lapl.i((.., ..5, ..))) - 5587.859016487346).abs() < 1e-6);
    assert!((fp(out_lapl.i((.., 5, ..))) - -783527.5368333755).abs() < 1e-4);
}

#[test]
fn get_rho_from_one_bra_mult_ket() {
    use rstsr::prelude::*;

    let mol_token = r#"
        atom = "C; H 1 0.94; H 1 0.94 2 104.5"
        basis = "def2-TZVP"
    "#;
    let coords = read_npz("ch2.npz", "coords").into_reverse_axes();
    let weights = read_npz("ch2.npz", "weights").into_reverse_axes();
    let ngrids = weights.shape()[0];
    let mo_coeff_full = read_npz("ch2.npz", "mo_coeff");
    let cint = CIntMol::from_toml(mol_token);
    let coords_array = coords.to_owned().into_pack_array::<3>(0).into_vec();
    let mut ni_obj = NIMatMul::new(&cint.cint, &coords_array, &weights.to_vec());

    // bra = alpha MOs (4 orbitals), ket_list = [alpha MOs, beta MOs]
    let bra = mo_coeff_full.i((0, .., ..4)).to_owned().into_contig(ColMajor);
    let ket_a = mo_coeff_full.i((0, .., ..4)).to_owned().into_contig(ColMajor);
    let ket_b = mo_coeff_full.i((1, .., ..4)).to_owned().into_contig(ColMajor);
    let ket_list = [ket_a.view(), ket_b.view()];

    // rho
    let out_rho = ni_obj.make_rho_from_one_bra_mult_ket(bra.view(), &ket_list, NIDenType::RHO).unwrap();
    assert_eq!(out_rho.shape(), &[ngrids, 1, 2]);
    assert!((fp(out_rho.view()) - 90.19801284927239).abs() < 1e-6);
    // sigma
    let out_sigma = ni_obj.make_rho_from_one_bra_mult_ket(bra.view(), &ket_list, NIDenType::SIGMA).unwrap();
    assert_eq!(out_sigma.shape(), &[ngrids, 4, 2]);
    assert!((fp(out_sigma.view()) - 358.2987825175244).abs() < 1e-6);
    // tau
    let out_tau = ni_obj.make_rho_from_one_bra_mult_ket(bra.view(), &ket_list, NIDenType::TAU).unwrap();
    assert_eq!(out_tau.shape(), &[ngrids, 5, 2]);
    assert!((fp(out_tau.view()) - 5531.545589708346).abs() < 1e-6);
    // lapl
    let out_lapl = ni_obj.make_rho_from_one_bra_mult_ket(bra.view(), &ket_list, NIDenType::LAPL).unwrap();
    assert_eq!(out_lapl.shape(), &[ngrids, 6, 2]);
    assert!((fp(out_lapl.i((.., ..5, ..))) - 5531.545589708346).abs() < 1e-6);
    assert!((fp(out_lapl.i((.., 5.., ..))) - -795547.4132924972).abs() < 1e-4);

    // Cross-check against make_rho_from_dm with symmetrized DMs
    let dm_a = bra.matmul(&ket_a.t());
    let dm_a_sym = (&dm_a + &dm_a.t()) * 0.5;
    let dm_b = bra.matmul(&ket_b.t());
    let dm_b_sym = (&dm_b + &dm_b.t()) * 0.5;
    for den_type in [NIDenType::RHO, NIDenType::SIGMA, NIDenType::TAU, NIDenType::LAPL] {
        let out = ni_obj.make_rho_from_one_bra_mult_ket(bra.view(), &ket_list, den_type).unwrap();
        let ref_a = ni_obj.make_rho_from_dm(&[dm_a_sym.view()], den_type).unwrap();
        let ref_b = ni_obj.make_rho_from_dm(&[dm_b_sym.view()], den_type).unwrap();
        let diff_a = (&out.i((.., .., 0)) - &ref_a.i((.., .., 0))).abs().max();
        let diff_b = (&out.i((.., .., 1)) - &ref_b.i((.., .., 0))).abs().max();
        assert!(diff_a < 1e-8, "Mismatch for alpha set, {:?}: max diff = {}", den_type, diff_a);
        assert!(diff_b < 1e-8, "Mismatch for beta set, {:?}: max diff = {}", den_type, diff_b);
    }
}

#[test]
fn get_rho_from_homogeneous_braket() {
    use rstsr::prelude::*;

    let mol_token = r#"
        atom = "C; H 1 0.94; H 1 0.94 2 104.5"
        basis = "def2-TZVP"
    "#;
    let coords = read_npz("ch2.npz", "coords").into_reverse_axes();
    let weights = read_npz("ch2.npz", "weights").into_reverse_axes();
    let ngrids = weights.shape()[0];
    let mo_coeff_full = read_npz("ch2.npz", "mo_coeff");
    let mo_occ_full = read_npz("ch2.npz", "mo_occ");
    let cint = CIntMol::from_toml(mol_token);
    // change coord Vec<f64> to Vec<[f64; 3]>
    let coords_array = coords.to_owned().into_pack_array::<3>(0).into_vec();
    let mut ni_obj = NIMatMul::new(&cint.cint, &coords_array, &weights.to_vec());
    // generate homogeneous braket for alpha and beta spin channels
    let mo_coeff_a = mo_coeff_full.i((0, .., ..)).to_owned().into_contig(ColMajor);
    let mo_occ_a = mo_occ_full.i((0, ..));
    let occ_mask_a = mo_occ_a.view().greater(0.0).into_vec();
    let occ_a = mo_occ_a.bool_select(0, &occ_mask_a);
    let bra_a = mo_coeff_a.bool_select(1, &occ_mask_a) * occ_a.sqrt().i((None, ..));

    let mo_coeff_b = mo_coeff_full.i((1, .., ..)).to_owned().into_contig(ColMajor);
    let mo_occ_b = mo_occ_full.i((1, ..));
    let occ_mask_b = mo_occ_b.view().greater(0.0).into_vec();
    let occ_b = mo_occ_b.bool_select(0, &occ_mask_b);
    let bra_b = mo_coeff_b.bool_select(1, &occ_mask_b) * occ_b.sqrt().i((None, ..));

    let bra_list = [bra_a.view(), bra_b.view()];

    // rho
    let out_rho = ni_obj.make_rho_from_homogeneous_braket(&bra_list, NIDenType::RHO).unwrap();
    assert_eq!(out_rho.shape(), &[ngrids, 1, 2]);
    assert!((fp(out_rho.view()) - 90.1600267407401).abs() < 1e-6);
    // sigma
    let out_sigma = ni_obj.make_rho_from_homogeneous_braket(&bra_list, NIDenType::SIGMA).unwrap();
    assert_eq!(out_sigma.shape(), &[ngrids, 4, 2]);
    assert!((fp(out_sigma.view()) - 369.8178428546338).abs() < 1e-6);
    // tau
    let out_tau = ni_obj.make_rho_from_homogeneous_braket(&bra_list, NIDenType::TAU).unwrap();
    assert_eq!(out_tau.shape(), &[ngrids, 5, 2]);
    assert!((fp(out_tau.view()) - 5587.859016487346).abs() < 1e-6);
    // lapl
    let out_lapl = ni_obj.make_rho_from_homogeneous_braket(&bra_list, NIDenType::LAPL).unwrap();
    assert_eq!(out_lapl.shape(), &[ngrids, 6, 2]);
    assert!((fp(out_lapl.i((.., ..5, ..))) - 5587.859016487346).abs() < 1e-6);
    assert!((fp(out_lapl.i((.., 5, ..))) - -783527.5368333755).abs() < 1e-4);
}
