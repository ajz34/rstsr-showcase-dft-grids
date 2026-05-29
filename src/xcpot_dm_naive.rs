//! Naive implementation of getting the XC potential from the density matrix.

use crate::prelude::*;
use libxc::prelude::*;

use LibXCSpin::*;

/// Reference implementation that covers `dft.numint.nr_rks` usage.
///
/// Make sure input density matrix is symmetrized.
///
/// # Parameters
///
/// - `ni_obj`: The NIMatmul object containing the grid and integral information.
/// - `xc_func`: The LibXC functional to evaluate.
/// - `dm0`: The density matrix, shape `[nao, nao]`.
pub fn compute_rks_vxc_from_dm_naive(
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

/// Reference implementation that covers `dft.numint.nr_rks` usage, using a homogenous bra instead
/// of density matrix.
///
/// Note for PySCF, the homogenous bra is constructed by `lib.tag_array`, tag `mo_coeff` and
/// `mo_occ` to density matrix (`rdm1`).
///
/// # Parameters
///
/// - `ni_obj`: The NIMatmul object containing the grid and integral information.
/// - `xc_func`: The LibXC functional to evaluate.
/// - `bra`: The homogenous bra, shape `[nao, nocc]`, where `nocc` is the number of occupied
///   orbitals. Note we do not have distinguish `mo_coeff` and `mo_occ`, so usually `bra` is
///   constructed by `mo_coeff * sqrt(mo_occ)`. This also requires occupation number to be
///   non-negative.
pub fn compute_rks_vxc_from_homogenous_bra_naive(
    ni_obj: &NIMatmul,
    xc_func: &LibXCFunctional,
    bra: TsrView,
) -> Result<(f64, f64, Tsr), NIError> {
    let device = bra.device().clone();
    let den_type = determine_den_type(xc_func)?;
    if xc_func.spin() != Unpolarized {
        return Err(ni_error!("Only unpolarized functionals are supported in this function"));
    }
    let ngrids = ni_obj.weights.len();
    ni_check_shape!(bra.ndim(), 2, "dm0 must be 2-dim for unpolarized case")?;
    let nao = bra.shape()[0];

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
        let rho = ni_cur.make_rho_from_homogeneous_braket(&[bra.view()], den_type)?;
        let [exc_eff, vxc_eff] = libxc_eval_eff(xc_func, rho.i((.., .., 0)), 1, true)?.try_into().unwrap();
        nelec += (&weights * rho.i((.., 0))).sum();
        exc += (exc_eff * &weights * rho.i((.., 0))).sum();
        let vxc_batch = ni_cur.make_vxc_pot_with_eff(vxc_eff.view(), den_type, 0)?;
        vxc += vxc_batch;
    }

    Ok((nelec, exc, vxc))
}

/// Reference implementation that covers `dft.numint.nr_rks_fxc` usage.
///
/// Make sure input density matrix is symmetrized.
///
/// # Parameters
///
/// - `ni_obj`: The NIMatmul object containing the grid and integral information.
/// - `xc_func`: The LibXC functional to evaluate.
/// - `dm0`: The density matrix (usually SCF density), shape `[nao, nao]`.
/// - `dm1`: The perturbed density matrix (usually from response), each of shape `[nao, nao]`.
pub fn compute_rks_fxc_from_dm_naive(
    ni_obj: &NIMatmul,
    xc_func: &LibXCFunctional,
    dm0: TsrView,
    dm1_list: &[TsrView],
) -> Result<Tsr, NIError> {
    let device = dm0.device().clone();
    let den_type = determine_den_type(xc_func)?;
    if xc_func.spin() != Unpolarized {
        return Err(ni_error!("Only unpolarized functionals are supported in this function"));
    }
    let ngrids = ni_obj.weights.len();
    ni_check_shape!(dm0.ndim(), 2, "dm0 must be 2-dim for unpolarized case")?;
    let nao = dm0.shape()[0];
    ni_check_shape!(dm0.shape(), [nao, nao], "dm0 must be square with shape (nao, nao) for unpolarized case")?;
    for dm1 in dm1_list {
        ni_check_shape!(dm1.shape(), [nao, nao], "dm1 must have shape (nao, nao) for unpolarized case")?;
    }
    let nset = dm1_list.len();

    let nbatch = ni_obj.nbatch;

    let mut fxc = rt::zeros(([nao, nao, nset], &device));

    for start in (0..ngrids).step_by(nbatch) {
        let stop = (start + nbatch).min(ngrids);
        let coords = &ni_obj.coords[start..stop];
        let weights = &ni_obj.weights[start..stop];
        // create a new NIMatmul object for the current batch of grid points
        let mut ni_cur = NIMatmul::new(&ni_obj.cint, coords, weights);
        ni_cur.nchunk = ni_obj.nchunk;

        let rho0 = ni_cur.make_rho_from_dm(&[dm0.view()], den_type)?;
        let rho1 = ni_cur.make_rho_from_dm(dm1_list, den_type)?;
        let xc_eff = libxc_eval_eff(xc_func, rho0.i((.., .., 0)), 2, true)?;
        let fxc_eff = &xc_eff[2];
        let fxc_batch = ni_cur.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), den_type, 0)?;
        fxc += fxc_batch;
    }

    Ok(fxc)
}

/// Reference implementation that covers `dft.numint.nr_rks_fxc` usage, using homogenous bra instead
/// of density matrix for dm0, and one_bra_mult_ket instead of list of densities for dm1.
///
/// Note for PySCF, the homogenous bra is constructed by `lib.tag_array`, tag `mo_coeff` and
/// `mo_occ` to density matrix.
///
/// # Parameters
///
/// - `ni_obj`: The NIMatmul object containing the grid and integral information.
/// - `xc_func`: The LibXC functional to evaluate.
/// - `bra0`: The homogenous bra for the unperturbed density, shape `[nao, nocc]`, where `nocc` is
///   the number of occupied orbitals.
/// - `bra1`: The homogenous bra for the perturbed density, shape `[nao, nocc1]`, where `nocc1` is
///   the number of occupied orbitals in bra1. For perturbed density, `bra1` is usually the original
///   occupied orbitals (not the ), we usually have `nocc1 == nocc`, but we do not strictly require
///   it here.
pub fn compute_rks_fxc_from_braket_naive(
    ni_obj: &NIMatmul,
    xc_func: &LibXCFunctional,
    bra0: TsrView,
    bra1: TsrView,
    ket1_list: &[TsrView],
) -> Result<Tsr, NIError> {
    let device = bra0.device().clone();
    let den_type = determine_den_type(xc_func)?;
    if xc_func.spin() != Unpolarized {
        return Err(ni_error!("Only unpolarized functionals are supported in this function"));
    }
    let ngrids = ni_obj.weights.len();
    ni_check_shape!(bra0.ndim(), 2, "bra0 must be 2-dim for unpolarized case")?;
    let nao = bra0.shape()[0];
    ni_check_shape!(bra1.ndim(), 2, "bra1 must be 2-dim for unpolarized case")?;
    ni_check_shape!(bra1.shape()[0], nao, "bra1 must have shape (nao, nocc) for unpolarized case")?;
    let nocc1 = bra1.shape()[1];
    for ket1 in ket1_list {
        ni_check_shape!(ket1.shape(), [nao, nocc1], "ket1 must have shape (nao, nocc) for unpolarized case")?;
    }
    let nset = ket1_list.len();

    let nbatch = ni_obj.nbatch;

    let mut fxc = rt::zeros(([nao, nao, nset], &device));

    for start in (0..ngrids).step_by(nbatch) {
        let stop = (start + nbatch).min(ngrids);
        let coords = &ni_obj.coords[start..stop];
        let weights = &ni_obj.weights[start..stop];
        // create a new NIMatmul object for the current batch of grid points
        let mut ni_cur = NIMatmul::new(&ni_obj.cint, coords, weights);
        ni_cur.nchunk = ni_obj.nchunk;

        let rho0 = ni_cur.make_rho_from_homogeneous_braket(&[bra0.view()], den_type)?;
        let rho1 = ni_cur.make_rho_from_one_bra_mult_ket(bra1.view(), ket1_list, den_type)?;
        let xc_eff = libxc_eval_eff(xc_func, rho0.i((.., .., 0)), 2, true)?;
        let fxc_eff = &xc_eff[2];
        let fxc_batch = ni_cur.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), den_type, 0)?;
        fxc += fxc_batch;
    }
    Ok(fxc)
}

/// Reference implementation that covers `dft.numint.nr_uks` usage.
///
/// Make sure input density matrix is symmetrized.
///
/// # Parameters
///
/// - `ni_obj`: The NIMatmul object containing the grid and integral information.
/// - `xc_func`: The LibXC functional to evaluate.
/// - `dm0`: The density matrix, shape `[nao, nao, 2]`.
pub fn compute_uks_vxc_from_dm_naive(
    ni_obj: &NIMatmul,
    xc_func: &LibXCFunctional,
    dm0: TsrView,
) -> Result<(f64, f64, Tsr), NIError> {
    let device = dm0.device().clone();
    let den_type = determine_den_type(xc_func)?;
    if xc_func.spin() != Polarized {
        return Err(ni_error!("Only polarized functionals are supported in this function"));
    }
    let ngrids = ni_obj.weights.len();
    ni_check_shape!(dm0.ndim(), 3, "dm0 must be 3-dim for polarized case")?;
    let nao = dm0.shape()[0];
    ni_check_shape!(dm0.shape(), [nao, nao, 2], "dm0 must have shape (nao, nao, 2) for polarized case")?;

    let nbatch = ni_obj.nbatch;

    let mut nelec = 0.0;
    let mut exc = 0.0;
    let mut vxc = rt::zeros(([nao, nao, 2], &device));

    for start in (0..ngrids).step_by(nbatch) {
        let stop = (start + nbatch).min(ngrids);
        let coords = &ni_obj.coords[start..stop];
        let weights = &ni_obj.weights[start..stop];
        // create a new NIMatmul object for the current batch of grid points
        let mut ni_cur = NIMatmul::new(&ni_obj.cint, coords, weights);
        ni_cur.nchunk = ni_obj.nchunk;

        let weights = rt::asarray((weights, &device));
        let rho = ni_cur.make_rho_from_dm(&[dm0.view()], den_type)?;
        let rho_spin_sum = rho.i((.., .., 0)) + rho.i((.., .., 1));
        let [exc_eff, vxc_eff] = libxc_eval_eff(xc_func, rho.view(), 1, true)?.try_into().unwrap();
        nelec += (&weights * &rho_spin_sum).sum() + (&weights * &rho_spin_sum).sum();
        exc += (exc_eff * &weights * &rho_spin_sum).sum();
        let vxc_batch = ni_cur.make_vxc_pot_with_eff(vxc_eff.view(), den_type, 0)?;
        vxc += vxc_batch;
    }

    Ok((nelec, exc, vxc))
}
