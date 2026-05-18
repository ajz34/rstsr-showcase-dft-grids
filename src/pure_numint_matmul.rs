use crate::prelude::*;

pub fn get_rho_from_dm(ao: TsrView, dm: TsrView, deriv: usize) -> Tsr {
    // ao: [ngrids, nao, ncomp]
    // dm: [nao, nao, nset]
    // output: [ngrids, ncomp', nset] in column-major, where ncomp' depends on the type of density
    // (e.g., 1 for rho, 1+3 for grad rho, 1+3+1 for tau, 1+3+1+1 for lapl rho)
    // Note this function forces 3-d dimensionality.

    // force dimensions to be 3-d for easier handling
    assert!(dm.ndim() == 3, "Density matrix must be 3D");
    assert!(ao.ndim() == 3, "AO values must be 3D");

    // check sanity of shapes
    assert_eq!(dm.shape()[0], dm.shape()[1], "Density matrix must be square");
    assert_eq!(ao.shape()[1], dm.shape()[0], "AO dimension must match density matrix dimension");
    let nao = dm.shape()[0];
    let nset = dm.shape()[2];
    let ngrid = ao.shape()[0];

    // handle deriv = 0 (rho only)
    match deriv {
        0 => {
            assert!(ao.shape()[2] >= 1, "For rho, AO must have 1 component");
            // rho_g^A = sum_(u v) P_(u v)^A * ao_(g u) * ao_(g v)
            // 1. T1_(g u)^A = P_(u v)^A * ao_(g u)
            let t1 = &ao.i((.., .., 0)) % dm.reshape((nao, nao * nset));
            let t1 = t1.into_shape((ngrid, nao, nset));
            // 2. rho_g^A = T1_(g u)^A * ao_(g u)
            rt::vecdot(t1, ao.i((.., .., 0)), 1)
        },
        _ => panic!("Unsupported derivative order: {deriv}"),
    }
}
