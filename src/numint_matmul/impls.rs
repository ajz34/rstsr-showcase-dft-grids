use super::prelude::*;

pub const AO_DERIV_DIM: [usize; 5] = [1, 4, 10, 20, 35];

impl NIMatMul {
    pub fn new(cint: &CInt, coords: &[[f64; 3]], weights: &[f64]) -> Self {
        Self { cint: cint.clone(), coords: coords.to_vec(), weights: weights.to_vec(), cache_tensor: HashMap::new() }
    }

    pub fn prepare_ao(&self, deriv: usize) -> Tsr {
        let eval_name = format!("deriv{}", deriv);
        let (out, shape) = self.cint.eval_gto(&eval_name, &self.coords).into();
        let device = DeviceTsr::default();
        rt::asarray((out, shape, &device))
    }

    pub fn get_cached_ao<'a>(&'a mut self, deriv: usize) -> TsrView<'a> {
        assert!(
            deriv < AO_DERIV_DIM.len(),
            "Derivative order {deriv} is too high, max supported is {}",
            AO_DERIV_DIM.len() - 1
        );
        // determine the maximum ao deriv that have already been computed and cached
        let max_cached_deriv = self
            .cache_tensor
            .keys()
            .filter_map(|k| k.strip_prefix("ao_deriv").and_then(|s| s.parse::<usize>().ok()))
            .max();
        // if the requested deriv is already cached, return it
        if let Some(max_deriv) = max_cached_deriv
            && max_deriv >= deriv
        {
            let cache_key = format!("ao_deriv{}", deriv);
            return self.cache_tensor.get(&cache_key).unwrap().i((.., .., ..AO_DERIV_DIM[deriv]));
        }

        // otherwise, compute and cache all missing ao deriv up to the requested one
        let key = format!("ao_deriv{}", deriv);
        self.cache_tensor.insert(key.clone(), self.prepare_ao(deriv));
        self.cache_tensor.get(&key).unwrap().view()
    }

    pub fn make_rho_from_dm(&mut self, dm: TsrView, den_type: NIDenType) -> Result<Tsr, NIError> {
        // This function assumes input dm is already symmetric.
        // If not, the caller should manually symmetrize it before calling this function.
        let ao = self.get_cached_ao(den_type.num_ao_deriv());

        let ngrid = ao.shape()[0];
        let nao = ao.shape()[1];
        ni_check_shape!(dm.ndim() >= 2, "Density matrix must be at least 2D")?;
        ni_check_shape!(dm.shape()[0..2], [nao, nao], "Density matrix must be square and match AO dimension")?;

        // reshape the density matrix to 3-dim [nao, nao, nset]
        let shape_suffix = dm.shape()[2..].to_vec();
        let nset = shape_suffix.iter().product();
        let dm_reshaped = dm.reshape([nao, nao, nset]);

        // compute the output
        let out_shape = [ngrid, den_type.num_rho_components(), nset];
        let device = ao.device().clone();
        let mut out = rt::zeros((out_shape.f(), &device));
        let mut buf = vec![0.0; ngrid * nao];
        get_rho_from_dm_with_output(ao, dm_reshaped.view(), den_type, out.view_mut(), &mut buf)?;

        // reshape output to match the original shape
        let out_shape = [ngrid, den_type.num_rho_components()].iter().chain(shape_suffix.iter()).cloned().collect_vec();
        Ok(out.into_shape(out_shape))
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
    let ni_obj = NIMatMul::new(&cint_mol.cint, &coords, &weights);
    let gto_eval = ni_obj.prepare_ao(1);
    println!("GTO evaluation result: {:10.5?}", gto_eval);
    let gto_eval = ni_obj.prepare_ao(0);
    println!("GTO evaluation result: {:10.5?}", gto_eval);
}
