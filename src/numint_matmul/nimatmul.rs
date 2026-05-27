use super::prelude::*;

pub struct NIMatMul<'a> {
    pub cint: CInt,
    pub coords: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub cache_tensor: HashMap<String, TsrCow<'a, f64>>,
}

impl<'a> NIMatMul<'a> {
    pub fn new(cint: &CInt, coords: &[[f64; 3]], weights: &[f64]) -> Self {
        assert!(coords.len() == weights.len(), "Number of coordinates must match number of weights");
        Self { cint: cint.clone(), coords: coords.to_vec(), weights: weights.to_vec(), cache_tensor: HashMap::new() }
    }

    pub fn prepare_ao(&self, deriv: usize) -> Tsr {
        let eval_name = format!("deriv{}", deriv);
        let (out, shape) = self.cint.eval_gto(&eval_name, &self.coords).into();
        let device = DeviceTsr::default();
        rt::asarray((out, shape, &device))
    }

    pub fn get_cached_ao(&mut self, deriv: usize) -> TsrView<'_> {
        assert!(
            deriv < AO_DERIV_DIM.len(),
            "Derivative order {deriv} is too high, max supported is {}",
            AO_DERIV_DIM.len() - 1
        );
        // determine the maximum ao deriv that have already been computed and cached
        let filter_closure = |k: &String| k.strip_prefix("ao_deriv").and_then(|s| s.parse::<usize>().ok());
        let max_cached_deriv = self.cache_tensor.keys().filter_map(filter_closure).max();
        // if the requested deriv is already cached, return it
        if let Some(max_deriv) = max_cached_deriv {
            if max_deriv >= deriv {
                let cache_key = format!("ao_deriv{}", deriv);
                return self.cache_tensor.get(&cache_key).unwrap().i((.., .., ..AO_DERIV_DIM[deriv]));
            }
        }

        // otherwise, compute and cache all missing ao deriv up to the requested one
        let key = format!("ao_deriv{}", deriv);
        self.cache_tensor.insert(key.clone(), self.prepare_ao(deriv).into_cow());
        self.cache_tensor.get(&key).unwrap().view()
    }

    pub fn make_rho_from_dm(&mut self, dm_list: &[TsrView], den_type: NIDenType) -> Result<Tsr, NIError> {
        let ao = self.get_cached_ao(den_type.num_ao_deriv());

        let ngrids = ao.shape()[0];
        let nao = ao.shape()[1];
        for dm in dm_list {
            ni_check_shape!(dm.ndim(), 2, "Each density matrix must be 2D")?;
            ni_check_shape!(dm.shape()[0..2], [nao, nao], "Density matrix must match AO dimension")?;
        }
        let nset = dm_list.len();

        let out_shape = [ngrids, den_type.num_nvar(), nset];
        let device = ao.device().clone();
        let mut out = rt::zeros((out_shape.f(), &device));
        let mut buf = vec![0.0; ngrids * nao];
        get_rho_from_dm_with_output(ao, dm_list, den_type, out.view_mut(), &mut buf)?;
        Ok(out)
    }

    pub fn make_rho_from_homogeneous_braket(
        &mut self,
        bra_list: &[TsrView],
        den_type: NIDenType,
    ) -> Result<Tsr, NIError> {
        let ao = self.get_cached_ao(den_type.num_ao_deriv());

        let ngrid = ao.shape()[0];
        let nao = ao.shape()[1];
        for bra in bra_list {
            ni_check_shape!(bra.ndim(), 2, "Each braket must be 2D")?;
            ni_check_shape!(bra.shape()[0], nao, "Bra's first dimension must match AO dimension")?;
        }
        let nocc_max = bra_list.iter().map(|bra| bra.shape()[1]).max().unwrap_or(0);
        let nset = bra_list.len();

        // compute the output
        let out_shape = [ngrid, den_type.num_nvar(), nset];
        let device = ao.device().clone();
        let mut out = rt::zeros((out_shape.f(), &device));
        let mut buf = vec![0.0; 2 * ngrid * nocc_max];
        get_rho_from_homogeneous_braket_with_output(ao, bra_list, den_type, out.view_mut(), &mut buf)?;
        Ok(out)
    }

    pub fn make_rho_from_one_bra_mult_ket(
        &mut self,
        bra: TsrView,
        ket_list: &[TsrView],
        den_type: NIDenType,
    ) -> Result<Tsr, NIError> {
        let ao = self.get_cached_ao(den_type.num_ao_deriv());

        let ngrid = ao.shape()[0];
        let nao = ao.shape()[1];
        ni_check_shape!(bra.ndim(), 2, "Bra must be 2D")?;
        ni_check_shape!(bra.shape()[0], nao, "Bra first dimension must match AO dimension")?;
        let nocc = bra.shape()[1];
        for ket in ket_list {
            ni_check_shape!(ket.ndim(), 2, "Each ket must be 2D")?;
            ni_check_shape!(ket.shape()[0], nao, "Ket first dimension must match AO dimension")?;
            ni_check_shape!(ket.shape()[1], nocc, "Ket second dimension must match bra")?;
        }
        let nset = ket_list.len();

        let out_shape = [ngrid, den_type.num_nvar(), nset];
        let device = ao.device().clone();
        let mut out = rt::zeros((out_shape.f(), &device));
        let mut buf = vec![0.0; 3 * ngrid * nocc];
        get_rho_from_one_bra_mult_ket_with_output(ao, bra, ket_list, den_type, out.view_mut(), &mut buf)?;
        Ok(out)
    }

    pub fn make_rho_from_mult_bra_mult_ket(
        &mut self,
        bra_list: &[TsrView],
        ket_list: &[TsrView],
        den_type: NIDenType,
    ) -> Result<Tsr, NIError> {
        let ao = self.get_cached_ao(den_type.num_ao_deriv());

        let ngrid = ao.shape()[0];
        let nao = ao.shape()[1];
        for (bra, ket) in bra_list.iter().zip(ket_list.iter()) {
            ni_check_shape!(bra.ndim(), 2, "Each bra must be 2D")?;
            ni_check_shape!(ket.ndim(), 2, "Each ket must be 2D")?;
            ni_check_shape!(nao, bra.shape()[0], "Bra first dimension must match AO dimension")?;
            ni_check_shape!(nao, ket.shape()[0], "Ket first dimension must match AO dimension")?;
            ni_check_shape!(bra.shape()[1], ket.shape()[1], "Bra and ket occupation must match")?;
        }
        let nocc_max = bra_list.iter().map(|bra| bra.shape()[1]).max().unwrap_or(0);
        let nset = bra_list.len();

        let out_shape = [ngrid, den_type.num_nvar(), nset];
        let device = ao.device().clone();
        let mut out = rt::zeros((out_shape.f(), &device));
        let mut buf = vec![0.0; 3 * ngrid * nocc_max];
        get_rho_from_mult_bra_mult_ket_with_output(ao, bra_list, ket_list, den_type, out.view_mut(), &mut buf)?;
        Ok(out)
    }

    pub fn make_vxc_pot_with_eff(
        &mut self,
        vxc_eff: TsrView,
        den_type: NIDenType,
        spin: usize,
    ) -> Result<Tsr, NIError> {
        let weights_data = self.weights.clone();
        let ao = self.get_cached_ao(den_type.num_ao_deriv());
        let nao = ao.shape()[1];
        let device = ao.device().clone();
        let weights_tsr = rt::asarray((weights_data.clone(), [weights_data.len()], &device));

        if spin == 0 {
            let mut out = rt::zeros(([nao, nao], &device));
            let mut buf = vec![0.0; weights_data.len() * nao];
            rks_vxc_pot_with_output(den_type, vxc_eff, ao, weights_tsr.view(), out.view_mut(), &mut buf)?;
            Ok(out)
        } else {
            let mut out = rt::zeros(([nao, nao, 2], &device));
            let mut buf = vec![0.0; weights_data.len() * nao];
            uks_vxc_pot_with_output(den_type, vxc_eff, ao, weights_tsr.view(), out.view_mut(), &mut buf)?;
            Ok(out)
        }
    }

    pub fn make_fxc_pot_with_eff(
        &mut self,
        fxc_eff: TsrView,
        rho1: TsrView,
        den_type: NIDenType,
        spin: usize,
    ) -> Result<Tsr, NIError> {
        let weights_data = self.weights.clone();
        let ao = self.get_cached_ao(den_type.num_ao_deriv());
        let nao = ao.shape()[1];
        let device = ao.device().clone();
        let weights_tsr = rt::asarray((weights_data.clone(), [weights_data.len()], &device));

        if spin == 0 {
            let nset = rho1.shape()[2];
            let mut out = rt::zeros(([nao, nao, nset], &device));
            let mut buf = vec![0.0; weights_data.len() * nao];
            rks_fxc_pot_with_output(den_type, fxc_eff, rho1, ao, weights_tsr.view(), out.view_mut(), &mut buf)?;
            Ok(out)
        } else {
            let nset = rho1.shape()[3];
            let mut out = rt::zeros(([nao, nao, 2, nset], &device));
            let mut buf = vec![0.0; weights_data.len() * nao];
            uks_fxc_pot_with_output(den_type, fxc_eff, rho1, ao, weights_tsr.view(), out.view_mut(), &mut buf)?;
            Ok(out)
        }
    }

    pub fn make_kxc_pot_with_eff(
        &mut self,
        kxc_eff: TsrView,
        rho1: TsrView,
        rho2: TsrView,
        den_type: NIDenType,
        spin: usize,
    ) -> Result<Tsr, NIError> {
        let weights_data = self.weights.clone();
        let ao = self.get_cached_ao(den_type.num_ao_deriv());
        let nao = ao.shape()[1];
        let device = ao.device().clone();
        let weights_tsr = rt::asarray((weights_data.clone(), [weights_data.len()], &device));

        if spin == 0 {
            let nset1 = rho1.shape()[2];
            let nset2 = rho2.shape()[2];
            let mut out = rt::zeros(([nao, nao, nset1, nset2], &device));
            let mut buf = vec![0.0; weights_data.len() * nao];
            rks_kxc_pot_with_output(den_type, kxc_eff, rho1, rho2, ao, weights_tsr.view(), out.view_mut(), &mut buf)?;
            Ok(out)
        } else {
            let nset1 = rho1.shape()[3];
            let nset2 = rho2.shape()[3];
            let mut out = rt::zeros(([nao, nao, 2, nset1, nset2], &device));
            let mut buf = vec![0.0; weights_data.len() * nao];
            uks_kxc_pot_with_output(den_type, kxc_eff, rho1, rho2, ao, weights_tsr.view(), out.view_mut(), &mut buf)?;
            Ok(out)
        }
    }
}
