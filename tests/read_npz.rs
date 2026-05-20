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
fn read_npz() {
    use npyz::NpyFile;
    use rstsr::prelude::*;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    let cargo_manifest_path = std::env!("CARGO_MANIFEST_DIR");
    let path = Path::new(cargo_manifest_path).join("tests").join("h2o.npz");
    let npz_file = BufReader::new(File::open(path).unwrap());
    let mut zip_file = zip::ZipArchive::new(npz_file).unwrap();
    let npy_file = zip_file.by_name("mo_coeff.npy").unwrap();
    let npy_reader = NpyFile::new(npy_file).unwrap();
    let shape = npy_reader.shape().iter().map(|&dim| dim as usize).collect::<Vec<_>>();
    let data = npy_reader.into_vec::<f64>().unwrap();
    let arr = rt::asarray((data, shape.c()));
    println!("MO coefficients:{:10.5?}", arr);
}
