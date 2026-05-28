#![allow(dead_code)]

use rstsr::prelude::*;

pub type Tsr<T = f64> = Tensor<T, DeviceFaer, IxD>;
pub type TsrView<'a, T = f64> = TensorView<'a, T, DeviceFaer, IxD>;

pub fn read_npz(file: &str, name: &str) -> Tsr<f64> {
    use npyz::NpyFile;
    use rstsr::prelude::*;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    let cargo_manifest_path = std::env!("CARGO_MANIFEST_DIR");
    let path = Path::new(cargo_manifest_path).join("benches").join(file);
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
    use rayon::prelude::*;
    x.iter_with_order(TensorIterOrder::F).into_par_iter().enumerate().map(|(i, &v)| (i as f64).cos() * v).sum()
}

#[macro_export]
macro_rules! fp_assert_eq {
    ($x:expr, $expected:expr, $tol:expr) => {{
        let diff = ($crate::util::fp($x) - $expected).abs();
        assert!(diff < $tol, "fp mismatch: diff {:.3e} > tol {:.3e}", diff, $tol);
    }};
}
