use super::prelude::*;

pub fn get_rho_from_dm_with_output(
    ao: TsrView,
    dm: TsrView,
    xctype: NIXCType,
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

    ni_check_shape!(out.shape().clone(), [ngrid, xctype.num_rho_components(), nset], "Output shape mismatch")?;
    ni_check_shape!(buf.len() >= ngrid * nao, "Buffer length insufficient")?;
    match xctype {
        LDA => ni_check_shape!(ao.shape()[2] >= 1, "AO at least 1 component (rho)")?,
        GGA | MGGA => ni_check_shape!(ao.shape()[2] >= 4, "AO at least 4 components (rho, x, y, z)")?,
        LAPL => ni_check_shape!(ao.shape()[2] >= 10, "AO at least 10 components")?,
    }

    let mut scratch = rt::asarray((buf, [ngrid, nao].f(), &device));

    for iset in 0..nset {
        // lda part
        scratch.matmul_from(ao.i((.., .., 0)), dm.i((.., .., iset)), 1.0, 0.0);
        out.i_mut((.., 0, iset)).vecdot_from(&scratch, ao.i((.., .., 0)), 1);
        // grad rho part
        if matches!(xctype, GGA | MGGA | LAPL) {
            out.i_mut((.., 1..4, iset)).vecdot_from(&scratch.i((.., .., None)), ao.i((.., .., 1..4)), 1);
            *&mut out.i_mut((.., 1..4, iset)) *= 2.0;
        }
        // lapl part (second derivative of AO)
        if matches!(xctype, LAPL) {
            for t in [4, 7, 9] {
                *&mut out.i_mut((.., 5, iset)) += 2.0 * rt::vecdot(&scratch, ao.i((.., .., t)), 1);
            }
        }
        // tau part
        if matches!(xctype, MGGA | LAPL) {
            for t in 1..4 {
                scratch.matmul_from(ao.i((.., .., t)), dm.i((.., .., iset)), 0.5, 0.0);
                *&mut out.i_mut((.., 4, iset)) += rt::vecdot(&scratch, ao.i((.., .., t)), 1);
            }
        }
        // lapl part (tau contribution)
        if matches!(xctype, LAPL) {
            let tau_contrib = 4.0 * out.i((.., 4, iset));
            *&mut out.i_mut((.., 5, iset)) += tau_contrib;
        }
    }

    Ok(())
}
