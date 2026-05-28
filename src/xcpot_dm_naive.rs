//! Naive implementation of getting the XC potential from the density matrix.

use crate::prelude::*;
use libxc::prelude::*;

use LibXCSpin::*;

pub fn make_rks_xcpot_from_dm_naive(
    ni_obj: &NIMatmul,
    xc_func: &LibXCFunctional,
    dm0: TsrView,
) -> Result<(f64, f64, Tsr), NIError> {
    let device = dm0.device().clone();
    let den_type = determine_den_type(xc_func)?;
    if xc_func.spin() != Unpolarized {
        return Err(ni_error!("Only unpolarized functionals are supported in this function"));
    }
    let ngrids = ni_obj.weights.len();
    ni_check_shape!(dm0.ndim(), 2, "dm0 must be 2-dim for unpolarized case")?;
    let nao = dm0.shape()[0];
    ni_check_shape!(dm0.shape(), [nao, nao], "dm0 must be square with shape (nao, nao) for unpolarized case")?;

    let nbatch = ni_obj.nbatch;

    let mut nelec = 0.0;
    let mut exc = 0.0;
    let mut vxc = rt::zeros(([nao, nao], &device));

    for start in (0..ngrids).step_by(nbatch) {
        let stop = (start + nbatch).min(ngrids);
        let coords = &ni_obj.coords[start..stop];
        let weights = &ni_obj.weights[start..stop];
        // create a new NIMatmul object for the current batch of grid points
        let mut ni_cur = NIMatmul::new(&ni_obj.cint, coords, weights);
        ni_cur.nchunk = ni_obj.nchunk;

        let weights = rt::asarray((weights, &device));
        let rho = ni_cur.make_rho_from_dm(&[dm0.view()], den_type)?;
        let [exc_eff, vxc_eff] = libxc_eval_eff(xc_func, rho.i((.., .., 0)), 1, true)?.try_into().unwrap();
        nelec += (&weights * rho.i((.., 0))).sum();
        exc += (exc_eff * &weights * rho.i((.., 0))).sum();
        let vxc_batch = ni_cur.make_vxc_pot_with_eff(vxc_eff.view(), den_type, 0)?;
        vxc += vxc_batch;
    }

    Ok((nelec, exc, vxc))
}
