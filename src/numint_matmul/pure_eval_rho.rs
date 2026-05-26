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
/// - `buf` : scratch buffer of length at least `ngrids * nao`
#[doc = include_str!("docs/get_rho_from_dm_with_output.md")]
pub fn get_rho_from_dm_with_output(
    ao: TsrView,
    dm_list: &[TsrView],
    den_type: NIDenType,
    mut out: TsrMut,
    buf: &mut [f64],
) -> Result<(), NIError> {
    // ao: [ngrids, nao, ncomp]
    // dm_list: each element is [nao, nao]
    // output: [ngrids, ncomp', nset] in column-major

    ni_check_shape!(ao.ndim(), 3, "AO values must be 3D")?;
    let nao = ao.shape()[1];

    for dm in dm_list {
        ni_check_shape!(dm.ndim(), 2, "Each density matrix must be 2D")?;
        ni_check_shape!(dm.shape()[0..2], [nao, nao], "Density matrix must match AO dimension")?;
    }
    let nset = dm_list.len();
    let ngrids = ao.shape()[0];
    let device = ao.device().clone();

    ni_check_shape!(out.shape().clone(), [ngrids, den_type.num_nvar(), nset], "Output shape mismatch")?;
    ni_check_shape!(buf.len() >= ngrids * nao, "Buffer length insufficient")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;

    let mut scr = rt::asarray((buf, [ngrids, nao].f(), &device));

    for (iset, dm) in dm_list.iter().enumerate() {
        // rho part
        scr.matmul_from(ao.i((.., .., 0)), dm, 1.0, 0.0);
        // out.i_mut((.., 0, iset)).vecdot_from(&scr, ao.i((.., .., 0)), 1);
        out.i_mut((.., 0, iset)).assign(rt::vecdot(&scr, ao.i((.., .., 0)), 1));
        // sigma part
        if matches!(den_type, SIGMA | TAU | LAPL) {
            out.i_mut((.., 1..4, iset)).vecdot_from(&scr.i((.., .., None)), ao.i((.., .., 1..4)), 1);
            *&mut out.i_mut((.., 1..4, iset)) *= 2.0;
        }
        // lapl part (second derivative of AO)
        if matches!(den_type, LAPL) {
            for t in [4, 7, 9] {
                *&mut out.i_mut((.., 5, iset)) += 2.0 * rt::vecdot(&scr, ao.i((.., .., t)), 1);
            }
        }
        // tau part
        if matches!(den_type, TAU | LAPL) {
            for t in 1..4 {
                scr.matmul_from(ao.i((.., .., t)), dm, 0.5, 0.0);
                *&mut out.i_mut((.., 4, iset)) += rt::vecdot(&scr, ao.i((.., .., t)), 1);
            }
        }
        // lapl part (tau contribution)
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
/// # Arguments
///
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `bra_list` : orbital coefficient matrices, each of shape `[nao, nocc]`; one per set, the
///   occupation number `nocc` is not required to be the same to the whole set
/// - `den_type` : which density components to compute
/// - `out` : output buffer, shape `[ngrids, num_rho_comp, nset]`
/// - `buf` : scratch buffer of length at least `2 * ngrid * nocc_max`
#[doc = include_str!("docs/get_rho_from_homogeneous_braket_with_output.md")]
pub fn get_rho_from_homogeneous_braket_with_output(
    ao: TsrView,
    bra_list: &[TsrView],
    den_type: NIDenType,
    mut out: TsrMut,
    buf: &mut [f64],
) -> Result<(), NIError> {
    // ao: [ngrids, nao, ncomp]
    // bra: [nao, nocc, nset]
    // output: [ngrids, ncomp', nset]

    ni_check_shape!(ao.ndim(), 3, "AO values must be 3D")?;
    let nao = ao.shape()[1];

    // check sanity of shapes
    for bra in bra_list {
        ni_check_shape!(bra.ndim(), 2, "Each bra must be 2D")?;
        ni_check_shape!(nao, bra.shape()[0], "AO dimension must match braket dimension")?;
    }
    let nocc_max = bra_list.iter().map(|bra| bra.shape()[1]).max().unwrap_or(0);

    let nset = bra_list.len();
    let ngrid = ao.shape()[0];
    let device = ao.device().clone();

    ni_check_shape!(out.shape().clone(), [ngrid, den_type.num_nvar(), nset], "Output shape mismatch")?;
    ni_check_shape!(buf.len() >= 2 * ngrid * nocc_max, "Buffer length insufficient")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;

    for (iset, bra) in bra_list.iter().enumerate() {
        let nocc = bra.shape()[1];
        let (buf1, buf2) = buf.split_at_mut(ngrid * nocc_max);
        let mut scr1 = rt::asarray((buf1, [ngrid, nocc].f(), &device));
        let mut scr2 = rt::asarray((buf2, [ngrid, nocc].f(), &device));
        // rho part
        scr1.matmul_from(ao.i((.., .., 0)), bra, 1.0, 0.0);
        out.i_mut((.., 0, iset)).vecdot_from(&scr1, &scr1, 1);
        if matches!(den_type, SIGMA | TAU | LAPL) {
            for t in 1..4 {
                scr2.matmul_from(ao.i((.., .., t)), bra, 1.0, 0.0);
                // sigma part
                out.i_mut((.., t, iset)).vecdot_from(&scr1, &scr2, 1);
                *&mut out.i_mut((.., t, iset)) *= 2.;
                // tau part
                if matches!(den_type, TAU | LAPL) {
                    *&mut out.i_mut((.., 4, iset)) += 0.5 * rt::vecdot(&scr2, &scr2, 1);
                }
            }
        }
        if matches!(den_type, LAPL) {
            // lapl part (second derivative of AO)
            for t in [4, 7, 9] {
                scr2.matmul_from(ao.i((.., .., t)), bra, 1.0, 0.0);
                *&mut out.i_mut((.., 5, iset)) += 2.0 * rt::vecdot(&scr1, &scr2, 1);
            }
            // lapl part (tau contribution)
            let tau_contrib = 4.0 * out.i((.., 4, iset));
            *&mut out.i_mut((.., 5, iset)) += tau_contrib;
        }
    }

    Ok(())
}

/// Evaluate density from orbital coefficients with one shared bra and multiple kets,
/// without forming the density matrix.
///
/// # Arguments
///
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `bra` : shared orbital coefficient matrix, shape `[nao, nocc]`
/// - `ket_list` : orbital coefficient matrices for each set, each of shape `[nao, nocc]`; the
///   occupation number `nocc` must be the same as `bra`
/// - `den_type` : which density components to compute
/// - `out` : output buffer, shape `[ngrids, num_rho_comp, nset]`
/// - `buf` : scratch buffer of length at least `3 * ngrid * nocc`
#[doc = include_str!("docs/get_rho_from_one_bra_mult_ket_with_output.md")]
pub fn get_rho_from_one_bra_mult_ket_with_output(
    ao: TsrView,
    bra: TsrView,
    ket_list: &[TsrView],
    den_type: NIDenType,
    mut out: TsrMut,
    buf: &mut [f64],
) -> Result<(), NIError> {
    // ao: [ngrids, nao, ncomp]
    // bra: [nao, nocc]
    // ket_list: each [nao, nocc]
    // output: [ngrids, ncomp', nset]

    ni_check_shape!(ao.ndim(), 3, "AO values must be 3D")?;
    let nao = ao.shape()[1];

    ni_check_shape!(bra.ndim(), 2, "Bra must be 2D")?;
    ni_check_shape!(nao, bra.shape()[0], "Bra first dimension must match AO dimension")?;
    let nocc = bra.shape()[1];

    for ket in ket_list {
        ni_check_shape!(ket.ndim(), 2, "Each ket must be 2D")?;
        ni_check_shape!(nao, ket.shape()[0], "Ket first dimension must match AO dimension")?;
        ni_check_shape!(ket.shape()[1], nocc, "Ket second dimension must match bra")?;
    }
    let nset = ket_list.len();
    let ngrid = ao.shape()[0];
    let device = ao.device().clone();

    ni_check_shape!(out.shape().clone(), [ngrid, den_type.num_nvar(), nset], "Output shape mismatch")?;
    ni_check_shape!(buf.len() >= 3 * ngrid * nocc, "Buffer length insufficient")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;

    let (buf1, buf_rest) = buf.split_at_mut(ngrid * nocc);
    let (buf2, buf3) = buf_rest.split_at_mut(ngrid * nocc);
    let mut scr1 = rt::asarray((buf1, [ngrid, nocc].f(), &device));
    let mut scr2 = rt::asarray((buf2, [ngrid, nocc].f(), &device));
    let mut scr3 = rt::asarray((buf3, [ngrid, nocc].f(), &device));

    // Pre-compute scr1 = ao_0 @ bra (shared across all sets)
    scr1.matmul_from(ao.i((.., .., 0)), &bra, 1.0, 0.0);

    for (iset, ket) in ket_list.iter().enumerate() {
        // rho part
        scr2.matmul_from(ao.i((.., .., 0)), ket, 1.0, 0.0);
        out.i_mut((.., 0, iset)).vecdot_from(&scr1, &scr2, 1);

        // sigma part
        if matches!(den_type, SIGMA | TAU | LAPL) {
            for t in 1..4 {
                scr3.matmul_from(ao.i((.., .., t)), ket, 1.0, 0.0);
                out.i_mut((.., t, iset)).vecdot_from(&scr1, &scr3, 1);
                scr3.matmul_from(ao.i((.., .., t)), &bra, 1.0, 0.0);
                *&mut out.i_mut((.., t, iset)) += rt::vecdot(&scr3, &scr2, 1);
            }
        }

        // lapl part (second derivative of AO), must come before tau which overwrites scr2
        if matches!(den_type, LAPL) {
            for t in [4, 7, 9] {
                scr3.matmul_from(ao.i((.., .., t)), ket, 1.0, 0.0);
                *&mut out.i_mut((.., 5, iset)) += rt::vecdot(&scr1, &scr3, 1);
                scr3.matmul_from(ao.i((.., .., t)), &bra, 1.0, 0.0);
                *&mut out.i_mut((.., 5, iset)) += rt::vecdot(&scr3, &scr2, 1);
            }
        }

        // tau part (overwrites scr2, which is no longer needed for sigma/lapl)
        if matches!(den_type, TAU | LAPL) {
            for t in 1..4 {
                scr2.matmul_from(ao.i((.., .., t)), ket, 1.0, 0.0);
                scr3.matmul_from(ao.i((.., .., t)), &bra, 1.0, 0.0);
                *&mut out.i_mut((.., 4, iset)) += 0.5 * rt::vecdot(&scr3, &scr2, 1);
            }
        }

        // lapl part (tau contribution)
        if matches!(den_type, LAPL) {
            let tau_contrib = 4.0 * out.i((.., 4, iset));
            *&mut out.i_mut((.., 5, iset)) += tau_contrib;
        }
    }

    Ok(())
}

/// Evaluate density from multiple bra-ket pairs, without forming the density matrix.
///
/// # Arguments
///
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `bra_list` : orbital coefficient matrices for bra, each of shape `[nao, nocc_i]`; the
///   occupation number `nocc_i` is not required to be the same across sets, but must match the
///   corresponding ket
/// - `ket_list` : orbital coefficient matrices for ket, each of shape `[nao, nocc_i]`; must have
///   the same length as `bra_list`, and `nocc_i` must match `bra_list[i]`
/// - `den_type` : which density components to compute
/// - `out` : output buffer, shape `[ngrids, num_rho_comp, nset]`
/// - `buf` : scratch buffer of length at least `3 * ngrid * nocc_max`
pub fn get_rho_from_mult_bra_mult_ket_with_output(
    ao: TsrView,
    bra_list: &[TsrView],
    ket_list: &[TsrView],
    den_type: NIDenType,
    mut out: TsrMut,
    buf: &mut [f64],
) -> Result<(), NIError> {
    // ao: [ngrids, nao, ncomp]
    // bra_list: each [nao, nocc_i]
    // ket_list: each [nao, nocc_i]
    // output: [ngrids, ncomp', nset]

    ni_check_shape!(ao.ndim(), 3, "AO values must be 3D")?;
    let nao = ao.shape()[1];

    ni_check_shape!(bra_list.len(), ket_list.len(), "bra_list and ket_list must have same length")?;
    let nocc_max = bra_list.iter().map(|bra| bra.shape()[1]).max().unwrap_or(0);

    for (bra, ket) in bra_list.iter().zip(ket_list.iter()) {
        ni_check_shape!(bra.ndim(), 2, "Each bra must be 2D")?;
        ni_check_shape!(ket.ndim(), 2, "Each ket must be 2D")?;
        ni_check_shape!(nao, bra.shape()[0], "Bra first dimension must match AO dimension")?;
        ni_check_shape!(nao, ket.shape()[0], "Ket first dimension must match AO dimension")?;
        ni_check_shape!(bra.shape()[1], ket.shape()[1], "Bra and ket occupation must match")?;
    }
    let nset = bra_list.len();
    let ngrid = ao.shape()[0];
    let device = ao.device().clone();

    ni_check_shape!(out.shape().clone(), [ngrid, den_type.num_nvar(), nset], "Output shape mismatch")?;
    ni_check_shape!(buf.len() >= 3 * ngrid * nocc_max, "Buffer length insufficient")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;

    for (iset, (bra, ket)) in bra_list.iter().zip(ket_list.iter()).enumerate() {
        let nocc = bra.shape()[1];
        let (buf1, buf_rest) = buf.split_at_mut(ngrid * nocc_max);
        let (buf2, buf3) = buf_rest.split_at_mut(ngrid * nocc_max);
        let mut scr1 = rt::asarray((buf1, [ngrid, nocc].f(), &device));
        let mut scr2 = rt::asarray((buf2, [ngrid, nocc].f(), &device));
        let mut scr3 = rt::asarray((buf3, [ngrid, nocc].f(), &device));

        // rho part
        scr1.matmul_from(ao.i((.., .., 0)), bra, 1.0, 0.0);
        scr2.matmul_from(ao.i((.., .., 0)), ket, 1.0, 0.0);
        out.i_mut((.., 0, iset)).vecdot_from(&scr1, &scr2, 1);

        // sigma part
        if matches!(den_type, SIGMA | TAU | LAPL) {
            for t in 1..4 {
                scr3.matmul_from(ao.i((.., .., t)), ket, 1.0, 0.0);
                out.i_mut((.., t, iset)).vecdot_from(&scr1, &scr3, 1);
                scr3.matmul_from(ao.i((.., .., t)), bra, 1.0, 0.0);
                *&mut out.i_mut((.., t, iset)) += rt::vecdot(&scr3, &scr2, 1);
            }
        }

        // lapl part (second derivative of AO), must come before tau which overwrites scr1/scr2
        if matches!(den_type, LAPL) {
            for t in [4, 7, 9] {
                scr3.matmul_from(ao.i((.., .., t)), ket, 1.0, 0.0);
                *&mut out.i_mut((.., 5, iset)) += rt::vecdot(&scr1, &scr3, 1);
                scr3.matmul_from(ao.i((.., .., t)), bra, 1.0, 0.0);
                *&mut out.i_mut((.., 5, iset)) += rt::vecdot(&scr3, &scr2, 1);
            }
        }

        // tau part (overwrites scr1/scr2, which are no longer needed for sigma/lapl)
        if matches!(den_type, TAU | LAPL) {
            for t in 1..4 {
                scr1.matmul_from(ao.i((.., .., t)), bra, 1.0, 0.0);
                scr2.matmul_from(ao.i((.., .., t)), ket, 1.0, 0.0);
                *&mut out.i_mut((.., 4, iset)) += 0.5 * rt::vecdot(&scr1, &scr2, 1);
            }
        }

        // lapl part (tau contribution)
        if matches!(den_type, LAPL) {
            let tau_contrib = 4.0 * out.i((.., 4, iset));
            *&mut out.i_mut((.., 5, iset)) += tau_contrib;
        }
    }

    Ok(())
}
