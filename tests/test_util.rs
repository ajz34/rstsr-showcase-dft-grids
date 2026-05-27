#![allow(dead_code)]

use libcint::prelude::*;
use rayon::prelude::*;
use rstsr::prelude::*;
use rstsr_showcase_dft_grids::prelude::*;

pub type Tsr<T = f64> = Tensor<T, DeviceFaer, IxD>;
pub type TsrView<'a, T = f64> = TensorView<'a, T, DeviceFaer, IxD>;

pub fn read_npz(file: &str, name: &str) -> Tsr<f64> {
    use npyz::NpyFile;
    use rstsr::prelude::*;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    let cargo_manifest_path = std::env!("CARGO_MANIFEST_DIR");
    let path = Path::new(cargo_manifest_path).join("tests").join(file);
    let npz_file = BufReader::new(File::open(path).unwrap());
    let mut zip_file = zip::ZipArchive::new(npz_file).unwrap();
    let npy_file = zip_file.by_name(&format!("{name}.npy")).unwrap();
    let npy_reader = NpyFile::new(npy_file).unwrap();
    let shape = npy_reader.shape().iter().map(|&dim| dim as usize).collect::<Vec<_>>();
    let data = npy_reader.into_vec::<f64>().unwrap();
    rt::asarray((data, shape.c()))
}

pub fn fp(x: TsrView<f64>) -> f64 {
    // fingerprint of tensor
    x.iter_with_order(TensorIterOrder::F).into_par_iter().enumerate().map(|(i, &v)| (i as f64).cos() * v).sum()
}

// ---------------------------------------------------------------------------
// Molecule data structs — shared setup (≡ Python's setUpModule + globals)
// ---------------------------------------------------------------------------

pub struct Ch2Molecule {
    pub mol_token: &'static str,
    pub coords: Tsr,
    pub weights: Tsr,
    pub ngrids: usize,
    pub rdm1: Tsr,
    pub mo_coeff: Tsr,
    pub mo_occ: Tsr,
    cint_mol: CIntMol,
}

impl Ch2Molecule {
    pub fn load() -> Self {
        let coords = read_npz("ch2.npz", "coords").into_reverse_axes();
        let weights = read_npz("ch2.npz", "weights").into_reverse_axes();
        let ngrids = weights.shape()[0];
        let rdm1 = read_npz("ch2.npz", "rdm1").into_reverse_axes();
        let mo_coeff = read_npz("ch2.npz", "mo_coeff");
        let mo_occ = read_npz("ch2.npz", "mo_occ");
        let mol_token = r#"atom = "C; H 1 0.94; H 1 0.94 2 104.5"
basis = "def2-TZVP""#;
        let cint_mol = CIntMol::from_toml(mol_token);
        Self { mol_token, coords, weights, ngrids, rdm1, mo_coeff, mo_occ, cint_mol }
    }

    pub fn cint(&self) -> &CInt {
        &self.cint_mol.cint
    }

    pub fn build_ni_obj(&self) -> NIMatMul<'_> {
        let coords_array = self.coords.to_owned().into_pack_array::<3>(0).into_vec();
        NIMatMul::new(self.cint(), &coords_array, &self.weights.to_vec())
    }

    pub fn bra_list(&self) -> [Tsr; 2] {
        let mo_coeff_a = self.mo_coeff.i((0, .., ..)).to_owned().into_contig(ColMajor);
        let mo_occ_a = self.mo_occ.i((0, ..));
        let occ_mask_a = mo_occ_a.view().greater(0.0).into_vec();
        let occ_a = mo_occ_a.bool_select(0, &occ_mask_a);
        let bra_a = mo_coeff_a.bool_select(1, &occ_mask_a) * occ_a.sqrt().i((None, ..));

        let mo_coeff_b = self.mo_coeff.i((1, .., ..)).to_owned().into_contig(ColMajor);
        let mo_occ_b = self.mo_occ.i((1, ..));
        let occ_mask_b = mo_occ_b.view().greater(0.0).into_vec();
        let occ_b = mo_occ_b.bool_select(0, &occ_mask_b);
        let bra_b = mo_coeff_b.bool_select(1, &occ_mask_b) * occ_b.sqrt().i((None, ..));

        [bra_a, bra_b]
    }
}

pub struct H2OMolecule {
    pub mol_token: &'static str,
    pub coords: Tsr,
    pub weights: Tsr,
    pub ngrids: usize,
    pub rdm1: Tsr,
    pub mo_coeff: Tsr,
    pub mo_occ: Tsr,
    cint_mol: CIntMol,
}

impl H2OMolecule {
    pub fn load() -> Self {
        let coords = read_npz("h2o.npz", "coords").into_reverse_axes();
        let weights = read_npz("h2o.npz", "weights").into_reverse_axes();
        let ngrids = weights.shape()[0];
        let rdm1 = read_npz("h2o.npz", "rdm1").into_reverse_axes();
        let mo_coeff = read_npz("h2o.npz", "mo_coeff");
        let mo_occ = read_npz("h2o.npz", "mo_occ");
        let mol_token = r#"atom = "O; H 1 0.94; H 1 0.94 2 104.5"
basis = "def2-TZVP""#;
        let cint_mol = CIntMol::from_toml(mol_token);
        Self { mol_token, coords, weights, ngrids, rdm1, mo_coeff, mo_occ, cint_mol }
    }

    pub fn cint(&self) -> &CInt {
        &self.cint_mol.cint
    }

    pub fn build_ni_obj(&self) -> NIMatMul<'_> {
        let coords_array = self.coords.to_owned().into_pack_array::<3>(0).into_vec();
        NIMatMul::new(self.cint(), &coords_array, &self.weights.to_vec())
    }

    pub fn bra_list(&self) -> [Tsr; 1] {
        let mo_coeff_arr = self.mo_coeff.to_owned().into_contig(ColMajor);
        let occ_mask = self.mo_occ.view().greater(0.0).into_vec();
        let occ = self.mo_occ.bool_select(0, &occ_mask);
        let bra = mo_coeff_arr.bool_select(1, &occ_mask) * occ.sqrt().i((None, ..));
        [bra]
    }
}

// ---------------------------------------------------------------------------
// Convenience macros
// ---------------------------------------------------------------------------

#[macro_export]
macro_rules! fp_assert_eq {
    ($x:expr, $expected:expr, $tol:expr) => {{
        let diff = ($crate::test_util::fp($x) - $expected).abs();
        assert!(diff < $tol, "fp mismatch: diff {:.3e} > tol {:.3e}", diff, $tol);
    }};
}
