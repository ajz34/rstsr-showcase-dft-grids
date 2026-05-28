//! Benchmarks for DFT numerical integral for pentacene (C22H14), def2-TZVP.
//!
//! To perform this benchmark, you need to first run `pen-bench.ipynb` to generate the `pen.npz`
//! file containing the molecular data and reference results. Then you can run `cargo bench`.
//!
//! ## Efficiency benchmark result
//!
//! Tested on AMD Ryzen 9 9955HX (16 cores).
//!
//! Functional: HYB_MGGA_XC_TPSSH (meta-GGA, tau-only)
//!
//! | Framework      | Property | Time (s) |
//! | -------------- | -------- | -------- |
//! | PySCF, NumInt  | vxc      |  6.94    |
//! |                | fxc      | 29.3     |
//! | Rust, NIMatmul | vxc      |  5.762   |
//! |                | fxc      | 21.308   |
//!
//! Please also note PySCF is shipped by PyPI, not the fully optimized compiled version. So actually
//! PySCF may be faster than the above numbers, but the relative performance should be similar.

mod util;

use itertools::Itertools;
use libcint::prelude::*;
use libxc::prelude::*;
use rstsr::prelude::*;
use rstsr_showcase_dft_grids::prelude::*;
use std::sync::LazyLock;
use std::time::Instant;
use util::*;

use LibXCSpin::*;
use NIDenType::*;

type Tsr<T = f64> = Tensor<T, DeviceFaer, IxD>;

// ---------------------------------------------------------------------------
// Shared data
// ---------------------------------------------------------------------------

pub struct PenMolecule {
    pub mol_token: &'static str,
    pub coords: Tsr,
    pub weights: Tsr,
    pub ngrids: usize,
    pub rdm1: Tsr,
    pub mo_coeff: Tsr,
    pub mo_occ: Tsr,
    pub cint_mol: CIntMol,
}

impl PenMolecule {
    pub fn load() -> Self {
        let coords = read_npz("pen.npz", "coords").into_reverse_axes();
        let weights = read_npz("pen.npz", "weights").into_reverse_axes();
        let ngrids = weights.shape()[0];
        let rdm1 = read_npz("pen.npz", "rdm1").into_reverse_axes();
        let mo_coeff = read_npz("pen.npz", "mo_coeff");
        let mo_occ = read_npz("pen.npz", "mo_occ");
        let mol_token = r#"
atom = """
    C   -6.1218053484 -0.7171513386  0.0000000000
    C   -4.9442285958 -1.4113046519  0.0000000000
    C   -3.6803098659 -0.7276441672  0.0000000000
    C   -2.4688049693 -1.4084918967  0.0000000000
    C   -1.2270315983 -0.7284607452  0.0000000000
    C    0.0000000000 -1.4090846909  0.0000000000
    C    1.2270315983 -0.7284607452  0.0000000000
    C    2.4688049693 -1.4084918967  0.0000000000
    C    3.6803098659 -0.7276441672  0.0000000000
    C    4.9442285958 -1.4113046519  0.0000000000
    C    6.1218053484 -0.7171513386  0.0000000000
    C    6.1218053484  0.7171513386  0.0000000000
    C    4.9442285958  1.4113046519  0.0000000000
    C    3.6803098659  0.7276441672  0.0000000000
    C    2.4688049693  1.4084918967  0.0000000000
    C    1.2270315983  0.7284607452  0.0000000000
    C    0.0000000000  1.4090846909  0.0000000000
    C   -1.2270315983  0.7284607452  0.0000000000
    C   -2.4688049693  1.4084918967  0.0000000000
    C   -3.6803098659  0.7276441672  0.0000000000
    C   -4.9442285958  1.4113046519  0.0000000000
    C   -6.1218053484  0.7171513386  0.0000000000
    H   -7.0692917090 -1.2490690741  0.0000000000
    H   -4.9430735200 -2.4988605526  0.0000000000
    H   -2.4690554105 -2.4968374995  0.0000000000
    H    0.0000000000 -2.4973235097  0.0000000000
    H    2.4690554105 -2.4968374995  0.0000000000
    H    4.9430735200 -2.4988605526  0.0000000000
    H    7.0692917090 -1.2490690741  0.0000000000
    H    7.0692917090  1.2490690741  0.0000000000
    H    4.9430735200  2.4988605526  0.0000000000
    H    2.4690554105  2.4968374995  0.0000000000
    H    0.0000000000  2.4973235097  0.0000000000
    H   -2.4690554105  2.4968374995  0.0000000000
    H   -4.9430735200  2.4988605526  0.0000000000
    H   -7.0692917090  1.2490690741  0.0000000000
"""
basis = "def2-TZVP""#;
        let cint_mol = CIntMol::from_toml(mol_token);
        Self { mol_token, coords, weights, ngrids, rdm1, mo_coeff, mo_occ, cint_mol }
    }

    pub fn cint(&self) -> &CInt {
        &self.cint_mol.cint
    }

    pub fn bra_list(&self) -> [Tsr; 1] {
        let mo_coeff_arr = self.mo_coeff.to_owned().into_contig(ColMajor);
        let occ_mask = self.mo_occ.view().greater(0.0).into_vec();
        let occ = self.mo_occ.bool_select(0, &occ_mask);
        let bra = mo_coeff_arr.bool_select(1, &occ_mask) * occ.sqrt().i((None, ..));
        [bra]
    }
}

static PEN: LazyLock<PenMolecule> = LazyLock::new(PenMolecule::load);

pub struct PenPerturbedDM {
    pub dm1_flat: Tsr,
    pub ncomp1: usize,
}

impl PenPerturbedDM {
    pub fn load(pen: &PenMolecule) -> Self {
        let device = pen.rdm1.device().clone();
        let get_intor = |name: &str| {
            let (out, shape) = pen.cint().integrate(name, None, None).into();
            rt::asarray((out, shape, &device))
        };
        let dm1: Tsr = (get_intor("int1e_r") + get_intor("int1e_giao_irjxp")) * &pen.rdm1;
        let dm1: Tsr = 0.5 * (&dm1 + dm1.swapaxes(0, 1));
        let ncomp1 = dm1.shape()[dm1.ndim() - 1];

        PenPerturbedDM { dm1_flat: dm1, ncomp1 }
    }
}

static PERTURBED_DM: LazyLock<PenPerturbedDM> = LazyLock::new(|| PenPerturbedDM::load(&PEN));

// ---------------------------------------------------------------------------
// Benchmarks (single-run, manual timing)
// ---------------------------------------------------------------------------

const GRID_BATCH: usize = 384 * 64;

fn bench_batched_vxc(pen: &PenMolecule) {
    let coords = pen.coords.to_owned().into_pack_array::<3>(0).into_vec();
    let weights = pen.weights.to_owned().into_vec();

    let mo_coeff = pen.mo_coeff.to_owned().into_contig(ColMajor);
    let mo_occ = pen.mo_occ.view();
    let occ_mask = mo_occ.view().greater(0.0).into_vec();
    let occ = mo_occ.bool_select(0, &occ_mask);
    let bra = mo_coeff.bool_select(1, &occ_mask) * occ.sqrt().i((None, ..));
    let nao = pen.mo_coeff.shape()[0];
    let device = pen.coords.device().clone();

    let time = Instant::now();
    let mut exc = 0.0;
    let mut vxc: Tsr = rt::zeros(([nao, nao], &device));
    for start in (0..pen.ngrids).step_by(GRID_BATCH) {
        let stop = (start + GRID_BATCH).min(pen.ngrids);
        let coords = &coords[start..stop];
        let weights = &weights[start..stop];
        let mut ni_obj = NIMatmul::new(pen.cint(), coords, weights);

        let weights = rt::asarray((weights, &device));
        let rho0 = ni_obj.make_rho_from_homogeneous_braket(&[bra.view()], TAU).unwrap();
        let xc_func = LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSSH", Unpolarized);
        let [exc_eff, vxc_eff] = libxc_eval_eff(&xc_func, rho0.i((.., .., 0)), 1, true).unwrap().try_into().unwrap();
        let exc_batch = (exc_eff * weights * rho0.i((.., 0))).sum();
        exc += exc_batch;
        let vxc_batch = ni_obj.make_vxc_pot_with_eff(vxc_eff.view(), TAU, 0).unwrap();
        vxc += vxc_batch;
    }
    let elapsed = time.elapsed().as_secs_f64();

    // Verification
    assert!((exc - -112.94843155506005).abs() < 1e-4, "exc mismatch: {exc}");
    fp_assert_eq!(vxc.view(), 1.371738499479081, 1e-4);

    println!("exc:    {exc}");
    println!("vxc fp: {:?}", fp(vxc.view()));
    println!("Time:   {elapsed:.3} s");
}

fn bench_batched_fxc(pen: &PenMolecule) {
    let coords = pen.coords.to_owned().into_pack_array::<3>(0).into_vec();
    let weights = pen.weights.to_owned().into_vec();

    let mo_coeff = pen.mo_coeff.to_owned().into_contig(ColMajor);
    let mo_occ = pen.mo_occ.view();
    let occ_mask = mo_occ.view().greater(0.0).into_vec();
    let occ = mo_occ.bool_select(0, &occ_mask);
    let bra = mo_coeff.bool_select(1, &occ_mask) * occ.sqrt().i((None, ..));
    let dm1 = PERTURBED_DM.dm1_flat.view();
    let dm1_list = dm1.axes_iter(-1).collect_vec();
    let device = pen.coords.device().clone();

    let time = Instant::now();
    let mut fxc: Tsr = rt::zeros((dm1.shape().to_vec(), &device));
    for start in (0..pen.ngrids).step_by(GRID_BATCH) {
        let stop = (start + GRID_BATCH).min(pen.ngrids);
        let coords = &coords[start..stop];
        let weights = &weights[start..stop];
        let mut ni_obj = NIMatmul::new(pen.cint(), coords, weights);

        let rho0 = ni_obj.make_rho_from_homogeneous_braket(&[bra.view()], TAU).unwrap();
        let rho1 = ni_obj.make_rho_from_dm(&dm1_list, TAU).unwrap();
        let xc_func = LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSSH", Unpolarized);
        let [_, _, fxc_eff] = libxc_eval_eff(&xc_func, rho0.i((.., .., 0)), 2, true).unwrap().try_into().unwrap();
        let fxc_batch = ni_obj.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), TAU, 0).unwrap();
        fxc += fxc_batch;
    }
    let elapsed = time.elapsed().as_secs_f64();

    // Verification
    fp_assert_eq!(fxc.view(), -3.2229077357662654, 1e-4);

    println!("fxc fp: {:?}", fp(fxc.view()));
    println!("Time:   {elapsed:.3} s");
}

fn main() {
    let pen = &PEN;

    println!("========== batched_vxc ==========");
    bench_batched_vxc(pen);

    println!();

    println!("========== batched_fxc ==========");
    bench_batched_fxc(pen);
}
