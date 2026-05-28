use super::prelude::*;
use NIDenType::*;

/// Contract AO values with a weight vector to produce a symmetric matrix.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `wv` : weight vector, shape `[ngrids, nvar]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `out` : output buffer, shape `[nao, nao]`; must be zeroed before calling
fn contract_ao_wv_naive(den_type: NIDenType, wv: TsrView, ao: TsrView, mut out: TsrMut) -> Result<(), NIError> {
    ni_check_shape!(wv.ndim(), 2, "Weight vector must be 2-dim")?;
    let nvar = wv.shape()[1];
    let ngrids = wv.shape()[0];
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type")?;
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim")?;
    let nao = ao.shape()[1];
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;
    ni_check_shape!(out.shape(), [nao, nao], "Output shape mismatch")?;

    if den_type == LAPL {
        return Err(ni_error!("Contracting AO with LAPL density type is not supported"));
    }

    macro_rules! ao_ {
        [$idx:expr] => { ao.i((.., .., $idx)) };
    }
    macro_rules! wv_ {
        [$idx:expr] => { wv.i((.., $idx)) };
    }

    // RHO contribution: 0.5 * ao[0].T @ (wv[0] * ao[0])
    *&mut out += 0.5 * ao_![0].t() % (wv_![0] * ao_![0]);
    // SIGMA contribution
    if matches!(den_type, SIGMA | TAU) {
        *&mut out += ao_![1].t() % (wv_![1] * ao_![0]);
        *&mut out += ao_![2].t() % (wv_![2] * ao_![0]);
        *&mut out += ao_![3].t() % (wv_![3] * ao_![0]);
    }
    // TAU contribution
    if matches!(den_type, TAU) {
        *&mut out += 0.25 * ao_![1].t() % (wv_![4] * ao_![1]);
        *&mut out += 0.25 * ao_![2].t() % (wv_![4] * ao_![2]);
        *&mut out += 0.25 * ao_![3].t() % (wv_![4] * ao_![3]);
    }
    // Symmetrize: out += out.T
    let out_t = out.t().to_owned();
    *&mut out += out_t;
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
/// - `vxc` : output vxc, shape `[nao, nao]`; must be zeroed before calling
pub fn rks_vxc_pot_with_eff_with_output_naive(
    den_type: NIDenType,
    vxc_eff: TsrView,
    ao: TsrView,
    weights: TsrView,
    vxc: TsrMut,
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

    if den_type == LAPL {
        return Err(ni_error!("Contracting AO with LAPL density type is not supported"));
    }

    let vxc_contracted = weights * vxc_eff;
    contract_ao_wv_naive(den_type, vxc_contracted.view(), ao, vxc)
}

/// Evaluate XC potential (2nd order, RKS) with fxc_eff.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `fxc_eff` : effective XC kernel, shape `[ngrids, nvar, nvar]`
/// - `rho1` : first-order density response, shape `[ngrids, nvar, nset]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `fxc` : output fxc, shape `[nao, nao, nset]`; must be zeroed before calling
pub fn rks_fxc_pot_with_eff_with_output_naive(
    den_type: NIDenType,
    fxc_eff: TsrView,
    rho1: TsrView,
    ao: TsrView,
    weights: TsrView,
    mut fxc: TsrMut,
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

    if den_type == LAPL {
        return Err(ni_error!("Contracting AO with LAPL density type is not supported"));
    }

    let fxc_eff_weighted = &weights * &fxc_eff;
    for i in 0..nset {
        let fxc_contracted = (&fxc_eff_weighted * rho1.i((.., .., None, i))).sum_axes(1);
        contract_ao_wv_naive(den_type, fxc_contracted.view(), ao.view(), fxc.i_mut((.., .., i)))?;
    }
    Ok(())
}

/// Evaluate XC potential (3rd order, RKS) with kxc_eff.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `kxc_eff` : effective XC kernel, shape `[ngrids, nvar, nvar, nvar]`
/// - `rho1` : first-order density response, shape `[ngrids, nvar, nset1]`
/// - `rho2` : second-order density response, shape `[ngrids, nvar, nset2]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `kxc` : output kxc, shape `[nao, nao, nset1, nset2]`; must be zeroed before calling
#[allow(clippy::too_many_arguments)]
pub fn rks_kxc_pot_with_eff_with_output_naive(
    den_type: NIDenType,
    kxc_eff: TsrView,
    rho1: TsrView,
    rho2: TsrView,
    ao: TsrView,
    weights: TsrView,
    mut kxc: TsrMut,
) -> Result<(), NIError> {
    ni_check_shape!(rho1.ndim(), 3, "rho1 tensor must be 3-dim")?;
    ni_check_shape!(rho2.ndim(), 3, "rho2 tensor must be 3-dim")?;
    let nset1 = rho1.shape()[2];
    let nset2 = rho2.shape()[2];
    let nvar = rho1.shape()[1];
    let ngrids = rho1.shape()[0];
    ni_check_shape!(kxc_eff.shape(), [ngrids, nvar, nvar, nvar], "kxc_eff shape mismatch")?;
    ni_check_shape!(rho2.shape()[0..2], [ngrids, nvar], "rho2 shape mismatch")?;
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch")?;
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type")?;
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim")?;
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;
    let nao = ao.shape()[1];
    ni_check_shape!(kxc.shape(), [nao, nao, nset1, nset2], "Output shape mismatch")?;

    if den_type == LAPL {
        return Err(ni_error!("Contracting AO with LAPL density type is not supported"));
    }

    let kxc_eff_weighted = &weights * &kxc_eff;

    for i2 in 0..nset2 {
        for i1 in 0..nset1 {
            let rho1_slice = rho1.i((.., .., None, None, i1)); // [ngrids, nvar, 1, 1]
            let temp = (&kxc_eff_weighted * &rho1_slice).sum_axes(1); // [ngrids, nvar, nvar]
            let rho2_slice = rho2.i((.., .., None, i2)); // [ngrids, nvar, 1]
            let kxc_contracted = (&temp * &rho2_slice).sum_axes(1); // [ngrids, nvar]
            contract_ao_wv_naive(den_type, kxc_contracted.view(), ao.view(), kxc.i_mut((.., .., i1, i2)))?;
        }
    }

    Ok(())
}

/// Evaluate XC potential (1st order, UKS) with vxc_eff.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `vxc_eff` : effective XC potential, shape `[ngrids, nvar, 2]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `vxc` : output vxc, shape `[nao, nao, 2]`; must be zeroed before calling
pub fn uks_vxc_pot_with_eff_with_output_naive(
    den_type: NIDenType,
    vxc_eff: TsrView,
    ao: TsrView,
    weights: TsrView,
    mut vxc: TsrMut,
) -> Result<(), NIError> {
    ni_check_shape!(vxc_eff.ndim(), 3, "Effective potential must be 3-dim")?;
    let nvar = vxc_eff.shape()[1];
    let ngrids = vxc_eff.shape()[0];
    ni_check_shape!(vxc_eff.shape()[2], 2, "vxc_eff must have 2 spin channels")?;
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch")?;
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type")?;
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim")?;
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;
    let nao = ao.shape()[1];
    ni_check_shape!(vxc.shape(), [nao, nao, 2], "Output shape mismatch")?;

    if den_type == LAPL {
        return Err(ni_error!("Contracting AO with LAPL density type is not supported"));
    }

    for s in 0..2 {
        let vxc_contracted = &weights * vxc_eff.i((.., .., s));
        contract_ao_wv_naive(den_type, vxc_contracted.view(), ao.view(), vxc.i_mut((.., .., s)))?;
    }
    Ok(())
}

/// Evaluate XC potential (2nd order, UKS) with fxc_eff.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `fxc_eff` : effective XC kernel, shape `[ngrids, nvar, 2, nvar, 2]`
/// - `rho1` : first-order density response, shape `[ngrids, nvar, 2, nset]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `fxc` : output fxc, shape `[nao, nao, 2, nset]`; must be zeroed before calling
pub fn uks_fxc_pot_with_eff_with_output_naive(
    den_type: NIDenType,
    fxc_eff: TsrView,
    rho1: TsrView,
    ao: TsrView,
    weights: TsrView,
    mut fxc: TsrMut,
) -> Result<(), NIError> {
    ni_check_shape!(rho1.ndim(), 4, "rho1 tensor must be 4-dim")?;
    let nset = rho1.shape()[3];
    let nvar = rho1.shape()[1];
    let ngrids = rho1.shape()[0];
    ni_check_shape!(rho1.shape()[2], 2, "rho1 must have 2 spin channels")?;
    ni_check_shape!(fxc_eff.shape(), [ngrids, nvar, 2, nvar, 2], "fxc_eff shape mismatch")?;
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch")?;
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type")?;
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim")?;
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;
    let nao = ao.shape()[1];
    ni_check_shape!(fxc.shape(), [nao, nao, 2, nset], "Output shape mismatch")?;

    if den_type == LAPL {
        return Err(ni_error!("Contracting AO with LAPL density type is not supported"));
    }

    let fxc_eff_weighted = &weights * &fxc_eff;
    for i in 0..nset {
        for s in 0..2 {
            let fxc_contracted =
                (&fxc_eff_weighted.i((.., .., .., .., s)) * rho1.i((.., .., .., None, i))).sum_axes([1, 2]);
            contract_ao_wv_naive(den_type, fxc_contracted.view(), ao.view(), fxc.i_mut((.., .., s, i)))?;
        }
    }
    Ok(())
}

/// Evaluate XC potential (3rd order, UKS) with kxc_eff.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `kxc_eff` : effective XC kernel, shape `[ngrids, nvar, 2, nvar, 2, nvar, 2]`
/// - `rho1` : first-order density response, shape `[ngrids, nvar, 2, nset1]`
/// - `rho2` : second-order density response, shape `[ngrids, nvar, 2, nset2]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `kxc` : output kxc, shape `[nao, nao, 2, nset1, nset2]`; must be zeroed before calling
#[allow(clippy::too_many_arguments)]
pub fn uks_kxc_pot_with_eff_with_output_naive(
    den_type: NIDenType,
    kxc_eff: TsrView,
    rho1: TsrView,
    rho2: TsrView,
    ao: TsrView,
    weights: TsrView,
    mut kxc: TsrMut,
) -> Result<(), NIError> {
    ni_check_shape!(rho1.ndim(), 4, "rho1 tensor must be 4-dim")?;
    ni_check_shape!(rho2.ndim(), 4, "rho2 tensor must be 4-dim")?;
    let nset1 = rho1.shape()[3];
    let nset2 = rho2.shape()[3];
    let nvar = rho1.shape()[1];
    let ngrids = rho1.shape()[0];
    ni_check_shape!(rho1.shape()[2], 2, "rho1 must have 2 spin channels")?;
    ni_check_shape!(rho2.shape()[0..3], [ngrids, nvar, 2], "rho2 shape mismatch")?;
    ni_check_shape!(kxc_eff.shape(), [ngrids, nvar, 2, nvar, 2, nvar, 2], "kxc_eff shape mismatch")?;
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch")?;
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type")?;
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim")?;
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;
    let nao = ao.shape()[1];
    ni_check_shape!(kxc.shape(), [nao, nao, 2, nset1, nset2], "Output shape mismatch")?;

    if den_type == LAPL {
        return Err(ni_error!("Contracting AO with LAPL density type is not supported"));
    }

    let kxc_eff_weighted = &weights * &kxc_eff;
    for i2 in 0..nset2 {
        for i1 in 0..nset1 {
            for s in 0..2 {
                let kxc_slice = kxc_eff_weighted.i((.., .., .., .., .., .., s)); // [ngrids, nvar, 2, nvar, 2, nvar]
                let rho1_slice = rho1.i((.., .., .., None, None, None, i1)); // [ngrids, nvar, 2, 1, 1, 1]
                let temp = (&kxc_slice * &rho1_slice).sum_axes([1, 2]); // [ngrids, nvar, 2, nvar]

                let rho2_slice = rho2.i((.., .., .., None, i2)); // [ngrids, nvar, 2, 1]
                let kxc_contracted = (&temp * &rho2_slice).sum_axes([1, 2]); // [ngrids, nvar]
                contract_ao_wv_naive(den_type, kxc_contracted.view(), ao.view(), kxc.i_mut((.., .., s, i1, i2)))?;
            }
        }
    }
    Ok(())
}

/// Contract AO with wv for RHO/SIGMA/TAU, bra-transformed variant.
///
/// This produces an asymmetric `[nao, nocc]` output (no symmetrization needed).
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `wv` : weight vector, shape `[ngrids, nvar]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `ao_bra` : bra-transformed AO values, shape `[ngrids, nocc, ncomp]`
/// - `out` : output buffer, shape `[nao, nocc]`; must be zeroed before calling
fn contract_ao_wv_bra_naive(
    den_type: NIDenType,
    wv: TsrView,
    ao: TsrView,
    ao_bra: TsrView,
    mut out: TsrMut,
) -> Result<(), NIError> {
    ni_check_shape!(wv.ndim(), 2, "Weight vector must be 2-dim")?;
    let nvar = wv.shape()[1];
    let ngrids = wv.shape()[0];
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type")?;
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim")?;
    let nao = ao.shape()[1];
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;
    ni_check_shape!(ao_bra.ndim(), 3, "ao_bra tensor must be 3-dim")?;
    ni_check_shape!(ao_bra.shape()[0], ngrids, "ao_bra grids dimension mismatch")?;
    let nocc = ao_bra.shape()[1];
    ni_check_shape!(ao_bra.shape()[2] >= den_type.num_ao_comp(), "ao_bra component dimension insufficient")?;
    ni_check_shape!(out.shape(), [nao, nocc], "Output shape mismatch")?;

    if den_type == LAPL {
        return Err(ni_error!("Contracting AO with LAPL density type is not supported"));
    }

    macro_rules! ao_ {
        [$idx:expr] => { ao.i((.., .., $idx)) };
    }
    macro_rules! ao_bra_ {
        [$idx:expr] => { ao_bra.i((.., .., $idx)) };
    }
    macro_rules! wv_ {
        [$idx:expr] => { wv.i((.., $idx)) };
    }

    // RHO contribution (coefficient 1.0, not 0.5 — no symmetrization)
    *&mut out += ao_![0].t() % (wv_![0] * ao_bra_![0]);

    // SIGMA contribution: ao_bra[t]*wv[t]@ao[0].T + ao_bra[0]*wv[t]@ao[t].T
    if matches!(den_type, SIGMA | TAU) {
        for t in 1..4 {
            *&mut out += ao_![0].t() % (wv_![t] * ao_bra_![t]);
            *&mut out += ao_![t].t() % (wv_![t] * ao_bra_![0]);
        }
    }

    // TAU contribution (coefficient 0.5, not 0.25 — no symmetrization)
    if matches!(den_type, TAU) {
        for t in 1..4 {
            *&mut out += 0.5 * ao_![t].t() % (wv_![4] * ao_bra_![t]);
        }
    }

    Ok(())
}

/// Evaluate XC potential (2nd order, RKS) with fxc_eff, bra transformed.
///
/// Bra is usually the occupied orbital coefficient (row-major applied to $\mu$),
/// which can lower the computational cost.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `fxc_eff` : effective XC kernel, shape `[ngrids, nvar, nvar]`
/// - `rho1` : first-order density response, shape `[ngrids, nvar, nset]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `bra` : bra orbital coefficients, shape `[nao, nocc]`
/// - `fxc` : output fxc (bra transformed), shape `[nao, nocc, nset]`; must be zeroed before calling
#[allow(clippy::too_many_arguments)]
pub fn rks_fxc_pot_with_eff_bra_trans_naive(
    den_type: NIDenType,
    fxc_eff: TsrView,
    rho1: TsrView,
    ao: TsrView,
    weights: TsrView,
    bra: TsrView,
    mut fxc: TsrMut,
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
    ni_check_shape!(bra.ndim(), 2, "bra must be 2-dim")?;
    ni_check_shape!(bra.shape()[0], nao, "bra first dimension must match nao")?;
    let nocc = bra.shape()[1];
    ni_check_shape!(fxc.shape(), [nao, nocc, nset], "Output shape mismatch")?;

    if den_type == LAPL {
        return Err(ni_error!("Contracting AO with LAPL density type is not supported"));
    }

    // Pre-compute ao_bra: [ngrids, nocc, ncomp] = ao @ bra for each component
    let device = ao.device().clone();
    let ncomp = den_type.num_ao_comp();
    let ao_bra_data = vec![0.0; ngrids * nocc * ncomp];
    let mut ao_bra = rt::asarray((ao_bra_data, [ngrids, nocc, ncomp], &device));
    for c in 0..ncomp {
        ao_bra.i_mut((.., .., c)).matmul_from(ao.i((.., .., c)), &bra, 1.0, 0.0);
    }

    let fxc_eff_weighted = &weights * &fxc_eff;
    for i in 0..nset {
        let fxc_contracted = (&fxc_eff_weighted * rho1.i((.., .., None, i))).sum_axes(1);
        contract_ao_wv_bra_naive(den_type, fxc_contracted.view(), ao.view(), ao_bra.view(), fxc.i_mut((.., .., i)))?;
    }
    Ok(())
}
