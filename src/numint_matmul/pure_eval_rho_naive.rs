use super::prelude::*;

/// Evaluate density from density matrices.
///
/// # Important Notes
///
/// The input density matrices must be symmetric.
/// You need to symmetrize them yourself. We will not check for symmetry.
///
/// # Parameters
///
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `dm_list` : density matrices, each of shape `[nao, nao]`; one per set
/// - `den_type` : which density components to compute
/// - `out` : output buffer, shape `[ngrids, num_rho_comp, nset]`
#[doc = include_str!("docs/get_rho_from_dm_with_output.md")]
pub fn get_rho_from_dm_with_output_naive(
    ao: TsrView,
    dm_list: &[TsrView],
    den_type: NIDenType,
    mut out: TsrMut,
) -> Result<(), NIError> {
    ni_check_shape!(ao.ndim(), 3, "AO values must be 3-dim")?;
    let nao = ao.shape()[1];

    for dm in dm_list {
        ni_check_shape!(dm.ndim(), 2, "Each density matrix must be 2-dim")?;
        ni_check_shape!(dm.shape()[0..2], [nao, nao], "Density matrix must match AO dimension")?;
    }
    let nset = dm_list.len();
    let ngrids = ao.shape()[0];

    ni_check_shape!(out.shape().clone(), [ngrids, den_type.num_nvar(), nset], "Output shape mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;

    for (iset, dm) in dm_list.iter().enumerate() {
        // rho part: scr = ao_0 @ dm
        let scr = ao.i((.., .., 0)) % dm;
        *&mut out.i_mut((.., 0, iset)) += (&scr * ao.i((.., .., 0))).sum_axes(1);
        // sigma part: out[..., 1..4, iset] += 2 * (scr * ao[..., 1..4]).sum(nao)
        if matches!(den_type, SIGMA | TAU | LAPL) {
            *&mut out.i_mut((.., 1..4, iset)) += 2 * (&scr * ao.i((.., .., 1..4))).sum_axes(1);
        }
        // lapl part (second derivative of AO)
        if matches!(den_type, LAPL) {
            for t in [4, 7, 9] {
                *&mut out.i_mut((.., 5, iset)) += 2 * (&scr * ao.i((.., .., t))).sum_axes(1);
            }
        }
        // tau part: scr = 0.5 * ao_t @ dm, then += (scr * ao_t).sum(nao)
        if matches!(den_type, TAU | LAPL) {
            for t in 1..4 {
                let scr = 0.5 * ao.i((.., .., t)) % dm;
                *&mut out.i_mut((.., 4, iset)) += (&scr * ao.i((.., .., t))).sum_axes(1);
            }
        }
        // lapl part (tau contribution): out[..., 5, iset] += 4 * out[..., 4, iset]
        if matches!(den_type, LAPL) {
            let tau_contrib = 4.0 * out.i((.., 4, iset));
            *&mut out.i_mut((.., 5, iset)) += tau_contrib;
        }
    }

    Ok(())
}

/// Evaluate density from orbital coefficients ("bra" vectors), where bra and ket are same, without
/// forming the density matrix.
///
/// # Parameters
///
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `bra_list` : orbital coefficient matrices, each of shape `[nao, nocc]`; one per set, the
///   occupation number `nocc` is not required to be the same to the whole set
/// - `den_type` : which density components to compute
/// - `out` : output buffer, shape `[ngrids, num_rho_comp, nset]`
#[doc = include_str!("docs/get_rho_from_homogeneous_braket_with_output.md")]
pub fn get_rho_from_homogeneous_braket_with_output_naive(
    ao: TsrView,
    bra_list: &[TsrView],
    den_type: NIDenType,
    mut out: TsrMut,
) -> Result<(), NIError> {
    ni_check_shape!(ao.ndim(), 3, "AO values must be 3-dim")?;
    let nao = ao.shape()[1];

    for bra in bra_list {
        ni_check_shape!(bra.ndim(), 2, "Each bra must be 2-dim")?;
        ni_check_shape!(nao, bra.shape()[0], "AO dimension must match braket dimension")?;
    }
    let nset = bra_list.len();
    let ngrids = ao.shape()[0];

    ni_check_shape!(out.shape().clone(), [ngrids, den_type.num_nvar(), nset], "Output shape mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;

    for (iset, bra) in bra_list.iter().enumerate() {
        // rho part: scr1 = ao_0 @ bra → [ngrids, nocc]
        let scr1 = ao.i((.., .., 0)) % bra;
        *&mut out.i_mut((.., 0, iset)) += (&scr1 * &scr1).sum_axes(1);
        // sigma & tau parts
        if matches!(den_type, SIGMA | TAU | LAPL) {
            for t in 1..4 {
                let scr2 = ao.i((.., .., t)) % bra;
                // sigma: out[..., t, iset] += 2 * (scr1 * scr2).sum(nocc)
                *&mut out.i_mut((.., t, iset)) += 2 * (&scr1 * &scr2).sum_axes(1);
                // tau: out[..., 4, iset] += 0.5 * (scr2 * scr2).sum(nocc)
                if matches!(den_type, TAU | LAPL) {
                    *&mut out.i_mut((.., 4, iset)) += 0.5 * (&scr2 * &scr2).sum_axes(1);
                }
            }
        }
        // lapl part
        if matches!(den_type, LAPL) {
            // second derivative of AO
            for t in [4, 7, 9] {
                let scr2 = ao.i((.., .., t)) % bra;
                *&mut out.i_mut((.., 5, iset)) += 2 * (&scr1 * &scr2).sum_axes(1);
            }
            // tau contribution: out[..., 5, iset] += 4 * out[..., 4, iset]
            let tau_contrib = 4.0 * out.i((.., 4, iset));
            *&mut out.i_mut((.., 5, iset)) += tau_contrib;
        }
    }

    Ok(())
}

/// Evaluate density from orbital coefficients with one shared bra and multiple kets,
/// without forming the density matrix.
///
/// # Parameters
///
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `bra` : shared orbital coefficient matrix, shape `[nao, nocc]`
/// - `ket_list` : orbital coefficient matrices for each set, each of shape `[nao, nocc]`; the
///   occupation number `nocc` must be the same as `bra`
/// - `den_type` : which density components to compute
/// - `out` : output buffer, shape `[ngrids, num_rho_comp, nset]`
#[doc = include_str!("docs/get_rho_from_one_bra_mult_ket_with_output.md")]
pub fn get_rho_from_one_bra_mult_ket_with_output_naive(
    ao: TsrView,
    bra: TsrView,
    ket_list: &[TsrView],
    den_type: NIDenType,
    mut out: TsrMut,
) -> Result<(), NIError> {
    ni_check_shape!(ao.ndim(), 3, "AO values must be 3-dim")?;
    let nao = ao.shape()[1];

    ni_check_shape!(bra.ndim(), 2, "Bra must be 2-dim")?;
    ni_check_shape!(nao, bra.shape()[0], "Bra first dimension must match AO dimension")?;
    let nocc = bra.shape()[1];

    for ket in ket_list {
        ni_check_shape!(ket.ndim(), 2, "Each ket must be 2-dim")?;
        ni_check_shape!(nao, ket.shape()[0], "Ket first dimension must match AO dimension")?;
        ni_check_shape!(ket.shape()[1], nocc, "Ket second dimension must match bra")?;
    }
    let nset = ket_list.len();
    let ngrids = ao.shape()[0];

    ni_check_shape!(out.shape().clone(), [ngrids, den_type.num_nvar(), nset], "Output shape mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;

    // Pre-compute scr1 = ao_0 @ bra (shared across all sets)
    let scr1 = ao.i((.., .., 0)) % &bra;

    for (iset, ket) in ket_list.iter().enumerate() {
        // rho part: scr2 = ao_0 @ ket
        let scr2 = ao.i((.., .., 0)) % ket;
        *&mut out.i_mut((.., 0, iset)) += (&scr1 * &scr2).sum_axes(1);

        // sigma part: out[..., t, iset] += (scr1 * scr3_ket).sum(nocc) + (scr3_bra * scr2).sum(nocc)
        if matches!(den_type, SIGMA | TAU | LAPL) {
            for t in 1..4 {
                let scr3 = ao.i((.., .., t)) % ket;
                *&mut out.i_mut((.., t, iset)) += (&scr1 * &scr3).sum_axes(1);
                let scr3 = ao.i((.., .., t)) % &bra;
                *&mut out.i_mut((.., t, iset)) += (&scr3 * &scr2).sum_axes(1);
            }
        }

        // lapl part (second derivative of AO), must come before tau which overwrites scr2
        if matches!(den_type, LAPL) {
            for t in [4, 7, 9] {
                let scr3 = ao.i((.., .., t)) % ket;
                *&mut out.i_mut((.., 5, iset)) += (&scr1 * &scr3).sum_axes(1);
                let scr3 = ao.i((.., .., t)) % &bra;
                *&mut out.i_mut((.., 5, iset)) += (&scr3 * &scr2).sum_axes(1);
            }
        }

        // tau part: out[..., 4, iset] += 0.5 * (scr3_bra * scr2_ket).sum(nocc)
        if matches!(den_type, TAU | LAPL) {
            for t in 1..4 {
                let scr2 = ao.i((.., .., t)) % ket;
                let scr3 = ao.i((.., .., t)) % &bra;
                *&mut out.i_mut((.., 4, iset)) += 0.5 * (&scr3 * &scr2).sum_axes(1);
            }
        }

        // lapl part (tau contribution): out[..., 5, iset] += 4 * out[..., 4, iset]
        if matches!(den_type, LAPL) {
            let tau_contrib = 4.0 * out.i((.., 4, iset));
            *&mut out.i_mut((.., 5, iset)) += tau_contrib;
        }
    }

    Ok(())
}

/// Evaluate density from multiple bra-ket pairs, without forming the density matrix.
///
/// # Parameters
///
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `bra_list` : orbital coefficient matrices for bra, each of shape `[nao, nocc_i]`; the
///   occupation number `nocc_i` is not required to be the same across sets, but must match the
///   corresponding ket
/// - `ket_list` : orbital coefficient matrices for ket, each of shape `[nao, nocc_i]`; must have
///   the same length as `bra_list`, and `nocc_i` must match `bra_list[i]`
/// - `den_type` : which density components to compute
/// - `out` : output buffer, shape `[ngrids, num_rho_comp, nset]`
pub fn get_rho_from_mult_bra_mult_ket_with_output_naive(
    ao: TsrView,
    bra_list: &[TsrView],
    ket_list: &[TsrView],
    den_type: NIDenType,
    mut out: TsrMut,
) -> Result<(), NIError> {
    ni_check_shape!(ao.ndim(), 3, "AO values must be 3-dim")?;
    let nao = ao.shape()[1];

    ni_check_shape!(bra_list.len(), ket_list.len(), "bra_list and ket_list must have same length")?;

    for (bra, ket) in bra_list.iter().zip(ket_list.iter()) {
        ni_check_shape!(bra.ndim(), 2, "Each bra must be 2-dim")?;
        ni_check_shape!(ket.ndim(), 2, "Each ket must be 2-dim")?;
        ni_check_shape!(nao, bra.shape()[0], "Bra first dimension must match AO dimension")?;
        ni_check_shape!(nao, ket.shape()[0], "Ket first dimension must match AO dimension")?;
        ni_check_shape!(bra.shape()[1], ket.shape()[1], "Bra and ket occupation must match")?;
    }
    let nset = bra_list.len();
    let ngrids = ao.shape()[0];

    ni_check_shape!(out.shape().clone(), [ngrids, den_type.num_nvar(), nset], "Output shape mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;

    for (iset, (bra, ket)) in bra_list.iter().zip(ket_list.iter()).enumerate() {
        // rho part: scr1 = ao_0 @ bra, scr2 = ao_0 @ ket
        let scr1 = ao.i((.., .., 0)) % bra;
        let scr2 = ao.i((.., .., 0)) % ket;
        *&mut out.i_mut((.., 0, iset)) += (&scr1 * &scr2).sum_axes(1);

        // sigma part: out[..., t, iset] += (scr1 * scr3_ket).sum(nocc) + (scr3_bra * scr2).sum(nocc)
        if matches!(den_type, SIGMA | TAU | LAPL) {
            for t in 1..4 {
                let scr3 = ao.i((.., .., t)) % ket;
                *&mut out.i_mut((.., t, iset)) += (&scr1 * &scr3).sum_axes(1);
                let scr3 = ao.i((.., .., t)) % bra;
                *&mut out.i_mut((.., t, iset)) += (&scr3 * &scr2).sum_axes(1);
            }
        }

        // lapl part (second derivative of AO), must come before tau which overwrites scr1/scr2
        if matches!(den_type, LAPL) {
            for t in [4, 7, 9] {
                let scr3 = ao.i((.., .., t)) % ket;
                *&mut out.i_mut((.., 5, iset)) += (&scr1 * &scr3).sum_axes(1);
                let scr3 = ao.i((.., .., t)) % bra;
                *&mut out.i_mut((.., 5, iset)) += (&scr3 * &scr2).sum_axes(1);
            }
        }

        // tau part: out[..., 4, iset] += 0.5 * (scr1_bra * scr2_ket).sum(nocc)
        if matches!(den_type, TAU | LAPL) {
            for t in 1..4 {
                let scr1 = ao.i((.., .., t)) % bra;
                let scr2 = ao.i((.., .., t)) % ket;
                *&mut out.i_mut((.., 4, iset)) += 0.5 * (&scr1 * &scr2).sum_axes(1);
            }
        }

        // lapl part (tau contribution): out[..., 5, iset] += 4 * out[..., 4, iset]
        if matches!(den_type, LAPL) {
            let tau_contrib = 4.0 * out.i((.., 4, iset));
            *&mut out.i_mut((.., 5, iset)) += tau_contrib;
        }
    }

    Ok(())
}
