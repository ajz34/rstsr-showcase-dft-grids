use crate::prelude::*;

pub struct NumIntMatMul {
    pub cint: CInt,
    pub coords: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub cache_tensor: HashMap<String, Tensor<f64>>,
}

impl NumIntMatMul {
    pub fn new(cint: &CInt, coords: &[[f64; 3]], weights: &[f64]) -> Self {
        Self { cint: cint.clone(), coords: coords.to_vec(), weights: weights.to_vec(), cache_tensor: HashMap::new() }
    }

    pub fn prepare_gto(&self, deriv: usize) -> Tsr {
        let eval_name = format!("deriv{}", deriv);
        let (out, shape) = self.cint.eval_gto(&eval_name, &self.coords).into();
        let device = DeviceTsr::default();
        rt::asarray((out, shape, &device))
    }
}

#[test]
fn playground() {
    let toml_str = r#"
        atom = "O; H 1 0.94; H 1 0.94 2 104.5"
        basis = "def2-TZVP"
    "#;
    let cint_mol = CIntMol::from_toml(toml_str);
    let ngrids = 10;
    let coords = (0..ngrids).map(|i| [(i as f64).sin(), (i as f64).cos(), (i as f64).tanh()]).collect::<Vec<_>>();
    let weights = (0..ngrids).map(|i| (i as f64).sin().abs()).collect::<Vec<_>>();
    let ni_obj = NumIntMatMul::new(&cint_mol.cint, &coords, &weights);
    let gto_eval = ni_obj.prepare_gto(1);
    println!("GTO evaluation result: {:10.5?}", gto_eval.reverse_axes());
}
