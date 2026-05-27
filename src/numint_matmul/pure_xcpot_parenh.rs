//! XC Potential generation (parallel enhanced)

use super::prelude::*;
use NIDenType::*;

/// Contract AO values with a weight vector to produce a symmetric matrix.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `wv` : weight vector, shape `[ngrids, nvar]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `out` : output buffer, shape `[nao, nao]`
/// - `buf` : scratch buffer of length at least `ngrids * nao`
fn contract_ao_wv_without_symmetrize(
    den_type: NIDenType,
    wv: TsrView,
    ao: TsrView,
    mut out: TsrMut,
    buf: &mut [f64],
) -> Result<(), NIError> {
    ni_check_shape!(wv.ndim(), 2, "Weight vector must be 2-dim")?;
    let nvar = wv.shape()[1];
    let ngrids = wv.shape()[0];
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type")?;
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim")?;
    let nao = ao.shape()[1];
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;

    ni_check_shape!(out.shape(), [nao, nao], "Output shape mismatch")?;
    ni_check_shape!(buf.len() >= ngrids * nao, "Buffer length insufficient")?;

    if den_type == LAPL {
        return Err(ni_error!("Contracting AO with LAPL density type is not supported"));
    }

    // clean notation of slice, just for readability

    /// Macro for slicing AO tensor with the last index.
    /// Returns [ngrids, nao] view for the specified component index.
    macro_rules! ao_ {
        [$idx:expr] => {
            ao.i((.., .., $idx))
        };
    }
    /// Macro for slicing weight vector with the last index.
    /// Returns [ngrids] view for the specified component index.
    macro_rules! wv_ {
        [$idx:expr] => {
            wv.i((.., $idx))
        };
    }

    let device = out.device().clone();
    let mut scr = rt::asarray((buf, [ngrids, nao], &device));

    // RHO contribution
    // out += 0.5 * ao_![0].t() % (wv_![0] * ao_![0]);
    rt::mul_with_output(ao_![0], wv_![0], scr.view_mut());
    out.matmul_from(ao_![0].t(), scr.view(), 0.5, 0.0);
    // SIGMA contribution
    if matches!(den_type, SIGMA | TAU) {
        // out += ao_![1].t() % (wv_![1] * ao_![0]);
        rt::mul_with_output(ao_![0], wv_![1], scr.view_mut());
        out.matmul_from(ao_![1].t(), scr.view(), 1.0, 1.0);
        // out += ao_![2].t() % (wv_![2] * ao_![0]);
        rt::mul_with_output(ao_![0], wv_![2], scr.view_mut());
        out.matmul_from(ao_![2].t(), scr.view(), 1.0, 1.0);
        // out += ao_![3].t() % (wv_![3] * ao_![0]);
        rt::mul_with_output(ao_![0], wv_![3], scr.view_mut());
        out.matmul_from(ao_![3].t(), scr.view(), 1.0, 1.0);
    }
    // TAU contribution
    if matches!(den_type, TAU) {
        // out += 0.25 * ao_![1].t() % (wv_![4] * ao_![1]);
        rt::mul_with_output(ao_![1], wv_![4], scr.view_mut());
        out.matmul_from(ao_![1].t(), scr.view(), 0.25, 1.0);
        // out += 0.25 * ao_![2].t() % (wv_![4] * ao_![2]);
        rt::mul_with_output(ao_![2], wv_![4], scr.view_mut());
        out.matmul_from(ao_![2].t(), scr.view(), 0.25, 1.0);
        // out += 0.25 * ao_![3].t() % (wv_![4] * ao_![3]);
        rt::mul_with_output(ao_![3], wv_![4], scr.view_mut());
        out.matmul_from(ao_![3].t(), scr.view(), 0.25, 1.0);
    }
    Ok(())
}

/// Evaluate XC potential (1st order) with vxc_eff.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `vxc_eff` : effective potential for XC, shape `[ngrids, nvar]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `vxc` : output vxc, shape `[nao, nao]`
pub fn rks_vxc_pot_with_output_parenh(
    den_type: NIDenType,
    vxc_eff: TsrView,
    ao: TsrView,
    weights: TsrView,
    mut vxc: TsrMut,
) -> Result<(), NIError> {
    ni_check_shape!(vxc_eff.ndim(), 2, "Effective potential must be 2-dim")?;
    let nvar = vxc_eff.shape()[1];
    let ngrids = vxc_eff.shape()[0];
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch")?;
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type")?;
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim")?;
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch")?;
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient")?;
    let nao = ao.shape()[1];
    ni_check_shape!(vxc.shape(), [nao, nao], "Output shape mismatch")?;

    if den_type == LAPL {
        return Err(ni_error!("Contracting AO with LAPL density type is not supported"));
    }

    // vxc_eff contraction
    let vxc_contracted = weights * vxc_eff;

    // buffer creation
    const NGRIDS_CHUNK: usize = 384;
    let buffer_init = || vec![0.0; NGRIDS_CHUNK * nao];
    let buffer_pool = BufferPool::new(buffer_init);
    let vxc_init = || vec![0.0; nao * nao];
    let vxc_pool = BufferPool::new(vxc_init);

    // task numbers
    let ntask_grid = ngrids.div_ceil(NGRIDS_CHUNK);
    let ntask = ntask_grid;

    // atomic guard to avoid racing write
    let guard = Mutex::new(());

    (0..ntask).into_par_iter().try_for_each(|i| {
        // determine the grid chunk for this task
        let start = i * NGRIDS_CHUNK;
        let end = ((i + 1) * NGRIDS_CHUNK).min(ngrids);

        // get buffer from pool
        let mut buf = buffer_pool.get();
        let mut vxc_local = rt::asarray((vxc_pool.get(), [nao, nao], ao.device()));

        // perform actual evaulation
        let vxc_contracted_chunk = vxc_contracted.i((start..end, ..));
        let ao_chunk = ao.i((start..end, .., ..));
        contract_ao_wv_without_symmetrize(
            den_type,
            vxc_contracted_chunk.view(),
            ao_chunk.view(),
            vxc_local.view_mut(),
            &mut buf,
        )?;

        // write back with lock
        let lock = guard.lock().unwrap();
        let mut vxc = unsafe { vxc.force_mut() };
        *&mut vxc += &vxc_local;
        drop(lock);

        // return buffer to pool
        buffer_pool.put(buf);
        vxc_pool.put(vxc_local.into_shape(-1).into_raw());
        Ok(())
    })?;

    // finally symmetrize the output
    let vxc_buf = vxc.swapaxes(0, 1).to_owned();
    *&mut vxc += vxc_buf;
    Ok(())
}
