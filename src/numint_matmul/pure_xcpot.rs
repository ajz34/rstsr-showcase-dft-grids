use super::prelude::*;
use NIDenType::*;

/// Contract AO values with a weight vector to produce a symmetric matrix.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `wv` : weight vector, shape `[ngrids, nvar]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `out` : output buffer, shape `[nao, nao]`
/// - `buf` : scratch buffer of length at least `ngrids * nao`
///
/// Note this function does not use the scratch buffer `buf`, so probably there has some allocation
/// cost, but the code is very simple.
fn contract_ao_wv_without_buf(
    den_type: NIDenType,
    wv: TsrView,
    ao: TsrView,
    mut out: TsrMut,
    buf: &mut [f64],
) -> Result<(), NIError> {
    ni_check_shape!(wv.ndim(), 2, "Weight vector must be 2-dim")?;
    let nvar = wv.shape()[1];
    let ngrids = wv.shape()[0];
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type")?;
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim")?;
    let nao = ao.shape()[1];
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;

    ni_check_shape!(out.shape(), [nao, nao], "Output shape mismatch")?;
    ni_check_shape!(buf.len() >= ngrids * nao, "Buffer length insufficient")?;

    if den_type == LAPL {
        return Err(ni_error!("Contracting AO with LAPL density type is not supported"));
    }

    // clean notation of slice
    use std::ops::RangeFull;
    const IA: [(RangeFull, RangeFull, usize); 5] = [(.., .., 0), (.., .., 1), (.., .., 2), (.., .., 3), (.., .., 4)];
    const IW: [(RangeFull, usize); 5] = [(.., 0), (.., 1), (.., 2), (.., 3), (.., 4)];

    // RHO contribution
    out += 0.5 * ao.i(IA[0]).t() % (wv.i(IW[0]) * ao.i(IA[0]));
    // SIGMA contribution
    if matches!(den_type, SIGMA | TAU) {
        out += ao.i(IA[1]).t() % (wv.i(IW[1]) * ao.i(IA[0]));
        out += ao.i(IA[2]).t() % (wv.i(IW[2]) * ao.i(IA[0]));
        out += ao.i(IA[3]).t() % (wv.i(IW[3]) * ao.i(IA[0]));
    }
    // TAU contribution
    if matches!(den_type, TAU) {
        out += 0.25 * ao.i(IA[1]).t() % (wv.i(IW[4]) * ao.i(IA[1]));
        out += 0.25 * ao.i(IA[2]).t() % (wv.i(IW[4]) * ao.i(IA[2]));
        out += 0.25 * ao.i(IA[3]).t() % (wv.i(IW[4]) * ao.i(IA[3]));
    }
    let out_t = out.t().to_owned();
    out += &out_t;
    Ok(())
}

/// Evaluate XC potential (1st order) with vxc_eff.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `vxc_eff` : effective potential for XC, shape `[ngrids, nvar]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `vxc` : output vxc, shape `[nao, nao]`
/// - `buf` : scratch buffer of length at least `ngrids * nao`
pub fn rks_vxc_pot_with_output(
    den_type: NIDenType,
    vxc_eff: TsrView,
    ao: TsrView,
    weights: TsrView,
    vxc: TsrMut,
    buf: &mut [f64],
) -> Result<(), NIError> {
    ni_check_shape!(vxc_eff.ndim(), 2, "Effective potential must be 2-dim")?;
    let nvar = vxc_eff.shape()[1];
    let ngrids = vxc_eff.shape()[0];
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch")?;
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type")?;
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim")?;
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;
    let nao = ao.shape()[1];
    ni_check_shape!(vxc.shape(), [nao, nao], "Output shape mismatch")?;
    ni_check_shape!(buf.len() >= ngrids * nao, "Buffer length insufficient")?;

    if den_type == LAPL {
        return Err(ni_error!("Contracting AO with LAPL density type is not supported"));
    }

    let vxc_contracted = weights * vxc_eff;
    contract_ao_wv_without_buf(den_type, vxc_contracted.view(), ao, vxc, buf)
}

pub fn rks_fxc_pot_with_output(
    den_type: NIDenType,
    fxc_eff: TsrView,
    rho1: TsrView,
    ao: TsrView,
    weights: TsrView,
    mut fxc: TsrMut,
    buf: &mut [f64],
) -> Result<(), NIError> {
    ni_check_shape!(rho1.ndim(), 3, "rho1 tensor must be 3-dim")?;
    let nset = rho1.shape()[2];
    let nvar = rho1.shape()[1];
    let ngrids = rho1.shape()[0];
    ni_check_shape!(fxc_eff.shape(), [ngrids, nvar, nvar], "fxc_eff shape mismatch")?;
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch")?;
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type")?;
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim")?;
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;
    let nao = ao.shape()[1];
    ni_check_shape!(fxc.shape(), [nao, nao, nset], "Output shape mismatch")?;
    ni_check_shape!(buf.len() >= ngrids * nao, "Buffer length insufficient")?;

    if den_type == LAPL {
        return Err(ni_error!("Contracting AO with LAPL density type is not supported"));
    }

    let fxc_eff_weighted = &weights * &fxc_eff;
    for i in 0..nset {
        let fxc_contracted = (&fxc_eff_weighted * rho1.i((.., .., None, i))).sum_axes(1);
        contract_ao_wv_without_buf(den_type, fxc_contracted.view(), ao.view(), fxc.i_mut((.., .., i)), buf)?;
    }
    Ok(())
}
