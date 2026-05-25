use crate::prelude::*;
use crate::xceff::xc_deriv::{transform_xc_inner, xc_indices_transform};
use libxc::compute_cpu::LibXCCpuInput;
use libxc::prelude::*;

use LibXCSpin::*;
use NIDenType::*;

pub fn determine_den_type(xc_func: &LibXCFunctional) -> Result<NIDenType, NIError> {
    match xc_func.family() {
        LibXCFamily::LDA | LibXCFamily::HybLDA => Ok(RHO),
        LibXCFamily::GGA | LibXCFamily::HybGGA => Ok(SIGMA),
        LibXCFamily::MGGA | LibXCFamily::HybMGGA => {
            if xc_func.needs_laplacian() {
                Ok(LAPL)
            } else {
                Ok(TAU)
            }
        },
        _ => Err(ni_error!("Unsupported functional family: {:?}", xc_func.family())),
    }
}

pub fn libxc_eval_inner(
    xc_func: &LibXCFunctional,
    rho: TsrView,
    deriv: usize,
) -> Result<(Vec<f64>, LibXCOutputLayout), NIError> {
    // sanity check
    // rho must be either [ngrids, 1/4/5/6] or [ngrids, 1/4/5/6, 2]
    match xc_func.spin() {
        Unpolarized => ni_check_shape!(rho.ndim(), 2, "rho for unpolarized functionals must be a 2D tensor")?,
        Polarized => {
            ni_check_shape!(rho.ndim(), 3, "rho for polarized functionals must be a 3D tensor")?;
            ni_check_shape!(rho.shape()[2], 2, "rho for polarized functionals must have last dimension of size 2")?;
        },
    }
    // we do not support laplacian currently
    if xc_func.needs_laplacian() {
        return Err(ni_error!("Laplacian-dependent functionals are not supported yet"));
    }
    let den_type = determine_den_type(xc_func)?;
    ni_check_shape!(rho.shape()[1] >= den_type.num_rho_comp(), "Input density does not have enough components")?;
    let do_rho = matches!(den_type, RHO | SIGMA | TAU | LAPL);
    let do_sigma = matches!(den_type, SIGMA | TAU | LAPL);
    let do_tau = matches!(den_type, TAU | LAPL);

    if xc_func.spin() == Unpolarized {
        // build up owned components
        let xc_rho = do_rho.then(|| rho.i((.., 0)).to_vec());
        let xc_sigma = do_sigma.then(|| rt::vecdot(rho.i((.., 1..4)), rho.i((.., 1..4)), 1).into_vec());
        let xc_tau = do_tau.then(|| rho.i((.., 4)).to_vec());
        // construct xc_input
        let mut xc_input = LibXCCpuInput::new();
        for (key, value) in [("rho", xc_rho.as_ref()), ("sigma", xc_sigma.as_ref()), ("tau", xc_tau.as_ref())] {
            value.map(|v| xc_input.insert(key.to_string(), v.as_slice()));
        }
        xc_func.compute_xc(&xc_input, deriv).map_err(|e| ni_error!("LibXC compute_xc failed: {e}"))
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
        xc_func.compute_xc(&xc_input, deriv).map_err(|e| ni_error!("LibXC compute_xc failed: {e}"))
    }
}

// transpose inplace, input slice is [m, n] in column-major
fn transpose_inplace(slc: &mut [f64], m: usize, n: usize) {
    // currently it is a naive implementation
    assert!(slc.len() >= n * m);
    let mut buffer = vec![0.0; n * m];
    for j in 0..m {
        for i in 0..n {
            buffer[j * n + i] = slc[i * m + j];
        }
    }
    slc[..n * m].copy_from_slice(&buffer);
}

pub fn libxc_eval_eff(xc_func: &LibXCFunctional, rho: TsrView, deriv: usize) -> Result<Vec<Tsr>, NIError> {
    let den_type = determine_den_type(xc_func)?;
    let (mut xc_val, xc_layout) = libxc_eval_inner(xc_func, rho.view(), deriv)?;
    // inplace transpose the spin-related components
    if xc_func.spin() == Polarized {
        let ngrids = rho.shape()[0];
        for name in xc_layout.component_names() {
            let r = xc_layout.get(name).unwrap();
            if r.end - r.start != ngrids {
                let ncomp = (r.end - r.start) / ngrids;
                transpose_inplace(&mut xc_val[r], ncomp, ngrids);
            }
        }
    }
    let ngrids = rho.shape()[0];
    let xlen = xc_val.len() / ngrids;
    let xc_val = rt::asarray((xc_val, [ngrids, xlen].f()));
    let xc_val = xc_indices_transform(xc_val.view(), den_type, xc_func.spin(), deriv);
    (0..=deriv).map(|order| transform_xc_inner(rho.view(), xc_val.view(), den_type, xc_func.spin(), order)).collect()
}
