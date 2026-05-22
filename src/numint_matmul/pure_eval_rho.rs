use super::prelude::*;

pub fn get_rho_from_dm_with_output(
    ao: TsrView,
    dm: TsrView,
    den_type: NIDenType,
    mut out: TsrMut,
    buf: &mut [f64],
) -> Result<(), NIError> {
    // ao: [ngrids, nao, ncomp]
    // dm: [nao, nao, nset]
    // output: [ngrids, ncomp', nset] in column-major, where ncomp' depends on the type of density
    // (e.g., 1 for rho, 1+3 for grad rho, 1+3+1 for tau, 1+3+1+1 for lapl rho)
    // Note this function forces 3-d dimensionality.

    // force dimensions to be 3-d for easier handling
    ni_check_shape!(dm.ndim(), 3, "Density matrix must be 3D")?;
    ni_check_shape!(ao.ndim(), 3, "AO values must be 3D")?;

    // check sanity of shapes
    ni_check_shape!(dm.shape()[0], dm.shape()[1], "Density matrix must be square")?;
    ni_check_shape!(ao.shape()[1], dm.shape()[0], "AO dimension must match density matrix dimension")?;

    let nao = dm.shape()[0];
    let nset = dm.shape()[2];
    let ngrid = ao.shape()[0];
    let device = ao.device().clone();

    ni_check_shape!(out.shape().clone(), [ngrid, den_type.num_rho_comp(), nset], "Output shape mismatch")?;
    ni_check_shape!(buf.len() >= ngrid * nao, "Buffer length insufficient")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;

    let mut scr = rt::asarray((buf, [ngrid, nao].f(), &device));

    for iset in 0..nset {
        // rho part
        scr.matmul_from(ao.i((.., .., 0)), dm.i((.., .., iset)), 1.0, 0.0);
        out.i_mut((.., 0, iset)).vecdot_from(&scr, ao.i((.., .., 0)), 1);
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
                scr.matmul_from(ao.i((.., .., t)), dm.i((.., .., iset)), 0.5, 0.0);
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

    ni_check_shape!(out.shape().clone(), [ngrid, den_type.num_rho_comp(), nset], "Output shape mismatch")?;
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
