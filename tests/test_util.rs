use rayon::prelude::*;
use rstsr::prelude::*;

type Tsr<T> = Tensor<T, DeviceFaer, IxD>;
type TsrView<'a, T> = TensorView<'a, T, DeviceFaer, IxD>;

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

#[test]
fn it_works() {
    use libcint::prelude::*;
    let cint_mol = CIntMol::from_toml(
        r#"
        atom = "O; H 1 0.94; H 1 0.94 2 104.5"
        basis = "def2-TZVP"
    "#,
    );
    let cint = cint_mol.cint;
    println!("{:?}", cint);

    let (data, _shape) = cint.integrate("int1e_ovlp", None, None).into();
    println!("Overlap integrals: {:10.5?}", data);
}

#[test]
fn read_rdm1() {
    let mo_coeff = read_npz("h2o.npz", "mo_coeff");
    println!("mo_coeff: {mo_coeff:10.5?}");
    println!("mo_coeff fingerprint: {}", fp(mo_coeff.t()));
}
