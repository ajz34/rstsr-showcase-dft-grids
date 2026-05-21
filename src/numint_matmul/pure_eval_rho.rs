use super::prelude::*;

pub fn get_rho_from_dm_with_output(ao: TsrView, dm: TsrView, xctype: NIXCType) -> Result<Tsr, NIError> {
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

    // prepare output buffer
    let mut out = match xctype {
        NIXCType::LDA => rt::zeros([ngrid, 1, nset]),
        NIXCType::GGA => rt::zeros([ngrid, 4, nset]),  // rho, x, y, z
        NIXCType::MGGA => rt::zeros([ngrid, 5, nset]), // rho, x, y, z, tau
        NIXCType::LAPL => rt::zeros([ngrid, 6, nset]), // rho, x, y, z, tau, lapl
    };

    // handle deriv = 0 (rho only)
    match xctype {
        NIXCType::LDA => {
            ni_check_shape!(ao.shape()[2] >= 1, "For rho, AO must have 1 component")?;
            // rho_g^A = sum_(u v) P_(u v)^A * ao_(g u) * ao_(g v)
            for iset in 0..nset {
                // 1. T1_(g u)^A = P_(u v)^A * ao_(g u)
                let t1 = &ao.i((.., .., 0)) % dm.i((.., .., iset));
                // 2. rho_g^A = T1_(g u)^A * ao_(g u)
                rt::vecdot_from(out.i_mut((.., 0, iset)), &t1, &ao.i((.., .., 0)), 1);
            }
        },
        _ => panic!("Unsupported XC type: {xctype:?}"),
    }
    Ok(out)
}
