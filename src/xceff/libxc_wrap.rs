use libxc::{compute_cpu::LibXCCpuInput, prelude::*};

use crate::prelude::*;

pub fn libxc_eval_inner(
    xc_func: &LibXCFunctional,
    rho: TsrView,
    deriv: usize,
) -> Result<(Vec<f64>, LibXCOutputLayout), NIError> {
    // sanity check
    // rho must be either [ngrids, 1/4/5/6] or [ngrids, 1/4/5/6, 2]
    let spin_polarize = match rho.ndim() {
        2 => LibXCSpin::Unpolarized,
        3 => {
            if rho.shape()[2] != 2 {
                return Err(ni_error!(
                    "rho for spin polarized functionals must have shape [ngrids, 1/4/5/6, 2], got shape {:?}",
                    rho.shape()
                ));
            }
            LibXCSpin::Polarized
        },
        _ => {
            return Err(ni_error!(
                "rho must be either [ngrids, 1/4/5/6] or [ngrids, 1/4/5/6, 2], got shape {:?}",
                rho.shape()
            ));
        },
    };
    // we do not support laplacian currently
    if xc_func.needs_laplacian() {
        return Err(ni_error!("Laplacian-dependent functionals are not supported yet"));
    }
    let den_type = match xc_func.family() {
        LibXCFamily::LDA | LibXCFamily::HybLDA => {
            ni_check_shape!(rho.shape()[1] >= 1, "LDA functionals require at least rho component")?;
            NIDenType::RHO
        },
        LibXCFamily::GGA | LibXCFamily::HybGGA => {
            ni_check_shape!(rho.shape()[1] >= 4, "GGA functionals require at least rho and sigma components")?;
            NIDenType::SIGMA
        },
        LibXCFamily::MGGA | LibXCFamily::HybMGGA => {
            // TODO: if we will support laplacian in future, the following match should modify
            ni_check_shape!(rho.shape()[1] >= 5, "MGGA functionals require at least rho, sigma and tau components")?;
            NIDenType::TAU
        },
        _ => return Err(ni_error!("Unsupported functional family: {:?}", xc_func.family())),
    };
    let do_rho = matches!(den_type, RHO | SIGMA | TAU | LAPL);
    let do_sigma = matches!(den_type, SIGMA | TAU | LAPL);
    let do_tau = matches!(den_type, TAU | LAPL);

    if spin_polarize == LibXCSpin::Unpolarized {
        // build up owned components
        let xc_rho = do_rho.then(|| rho.i((.., 0)).to_vec());
        let xc_sigma = do_sigma.then(|| rt::vecdot(rho.i((.., 1..4)), rho.i((.., 1..4)), 1).into_vec());
        let xc_tau = do_tau.then(|| rho.i((.., 4)).to_vec());
        // construct xc_input
        let mut xc_input = LibXCCpuInput::new();
        for (key, value) in [("rho", xc_rho.as_ref()), ("sigma", xc_sigma.as_ref()), ("tau", xc_tau.as_ref())] {
            value.map(|v| xc_input.insert(key.to_string(), v.as_slice()));
        }
        xc_func.compute_xc(&xc_input, deriv).map_err(|e| ni_error!("LibXC compute_xc failed: {}", e))
    } else {
        // build up owned components
        // note libxc's convention is [2, ngrids] for rho, not [ngrids, 2].
        // we need to transpose the input rho before feeding into libxc.
        let xc_rho = do_rho.then(|| rho.i((.., 0, ..)).t().into_shape(-1).to_vec());
        let xc_tau = do_tau.then(|| rho.i((.., 4, ..)).t().into_shape(-1).to_vec());
        // sigma is more complicated, as it requires (uu ud dd) components.
        let xc_sigma = do_sigma.then(|| {
            let ngrids = rho.shape()[0];
            let mut sigma = rt::zeros(([ngrids, 3], &rho.device().clone()));
            sigma.i_mut((.., 0)).vecdot_from(rho.i((.., 1..4, 0)), rho.i((.., 1..4, 0)), 1);
            sigma.i_mut((.., 1)).vecdot_from(rho.i((.., 1..4, 0)), rho.i((.., 1..4, 1)), 1);
            sigma.i_mut((.., 2)).vecdot_from(rho.i((.., 1..4, 1)), rho.i((.., 1..4, 1)), 1);
            sigma.t().into_shape(-1).to_vec()
        });
        let mut xc_input = LibXCCpuInput::new();
        for (key, value) in [("rho", xc_rho.as_ref()), ("sigma", xc_sigma.as_ref()), ("tau", xc_tau.as_ref())] {
            value.map(|v| xc_input.insert(key.to_string(), v.as_slice()));
        }
        xc_func.compute_xc(&xc_input, deriv).map_err(|e| ni_error!("LibXC compute_xc failed: {}", e))
    }
}
