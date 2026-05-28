mod test_util;

use itertools::Itertools;
use libcint::prelude::*;
use libxc::prelude::*;
use rstest::*;
use rstsr::prelude::*;
use rstsr_showcase_dft_grids::prelude::*;
use test_util::*;

use LibXCSpin::*;
use NIDenType::*;

type Tsr<T = f64> = Tensor<T, DeviceFaer, IxD>;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

pub struct PenMolecule {
    pub mol_token: &'static str,
    pub coords: Tsr,
    pub weights: Tsr,
    pub ngrids: usize,
    pub rdm1: Tsr,
    pub mo_coeff: Tsr,
    pub mo_occ: Tsr,
    cint_mol: CIntMol,
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

    pub fn build_ni_obj(&self) -> NIMatmul<'_> {
        let coords_array = self.coords.to_owned().into_pack_array::<3>(0).into_vec();
        NIMatmul::new(self.cint(), &coords_array, &self.weights.to_vec())
    }

    pub fn bra_list(&self) -> [Tsr; 1] {
        let mo_coeff_arr = self.mo_coeff.to_owned().into_contig(ColMajor);
        let occ_mask = self.mo_occ.view().greater(0.0).into_vec();
        let occ = self.mo_occ.bool_select(0, &occ_mask);
        let bra = mo_coeff_arr.bool_select(1, &occ_mask) * occ.sqrt().i((None, ..));
        [bra]
    }
}

#[fixture]
#[once]
fn pen() -> PenMolecule {
    PenMolecule::load()
}

pub struct PenPerturbedDM {
    pub dm1_flat: Tsr,
    pub ncomp1: usize,
}

#[fixture]
#[once]
fn perturbed_dm(pen: &PenMolecule) -> PenPerturbedDM {
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

#[rstest]
fn batched_vxc(pen: &PenMolecule) {
    // this should recover the pyscf's similar usage of ni.nr_vxc.

    /* Benchmark result
       - AMD Ryzen 9 9955HX 16-Core Processor
       - PySCF: 6.94 sec
       - This implementation: 8.51 sec
    */

    const GRID_BATCH: usize = 384 * 64;

    // grid setting
    let coords = pen.coords.to_owned().into_pack_array::<3>(0).into_vec();
    let weights = pen.weights.to_owned().into_vec();

    // bra, dm1 setting
    let mo_coeff = pen.mo_coeff.to_owned().into_contig(ColMajor);
    let mo_occ = pen.mo_occ.view();
    let occ_mask = mo_occ.view().greater(0.0).into_vec();
    let occ = mo_occ.bool_select(0, &occ_mask);
    let bra = mo_coeff.bool_select(1, &occ_mask) * occ.sqrt().i((None, ..));
    let nao = pen.mo_coeff.shape()[0];
    let device = pen.coords.device().clone();

    let time = std::time::Instant::now();
    let mut exc = 0.0;
    let mut vxc: Tsr = rt::zeros(([nao, nao], &device));
    for start in (0..pen.ngrids).step_by(GRID_BATCH) {
        let stop = (start + GRID_BATCH).min(pen.ngrids);
        let coords = &coords[start..stop];
        let weights = &weights[start..stop];
        let mut ni_obj = NIMatmul::new(pen.cint(), coords, weights);

        // generate rho
        let weights = rt::asarray((weights, &device));
        let rho0 = ni_obj.make_rho_from_homogeneous_braket(&[bra.view()], TAU).unwrap();
        let xc_func = LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSSH", Unpolarized);
        let [exc_eff, vxc_eff] = libxc_eval_eff(&xc_func, rho0.i((.., .., 0)), 1, true).unwrap().try_into().unwrap();
        let exc_batch = (exc_eff * weights * rho0.i((.., 0))).sum();
        exc += exc_batch;
        let vxc_batch = ni_obj.make_vxc_pot_with_eff(vxc_eff.view(), TAU, 0).unwrap();
        vxc += vxc_batch;
    }
    println!("exc: {exc}");
    println!("vxc fp: {:?}", fp(vxc.view()));
    println!("Time taken: {:.3} seconds", time.elapsed().as_secs_f64());
    assert!((exc - -112.94843155506005).abs() < 1e-4);
    fp_assert_eq!(vxc.view(), 1.371738499479081, 1e-4);
}

#[rstest]
fn batched_fxc(pen: &PenMolecule, perturbed_dm: &PenPerturbedDM) {
    // this should recover the pyscf's similar usage of ni.nr_fxc.

    /* Benchmark result
       - AMD Ryzen 9 9955HX 16-Core Processor
       - PySCF: 29.3 sec
       - This implementation: 38.9 sec
    */

    const GRID_BATCH: usize = 384 * 64;

    // grid setting
    let coords = pen.coords.to_owned().into_pack_array::<3>(0).into_vec();
    let weights = pen.weights.to_owned().into_vec();

    // bra, dm1 setting
    let mo_coeff = pen.mo_coeff.to_owned().into_contig(ColMajor);
    let mo_occ = pen.mo_occ.view();
    let occ_mask = mo_occ.view().greater(0.0).into_vec();
    let occ = mo_occ.bool_select(0, &occ_mask);
    let bra = mo_coeff.bool_select(1, &occ_mask) * occ.sqrt().i((None, ..));
    let dm1 = perturbed_dm.dm1_flat.view();
    let dm1_list = dm1.axes_iter(-1).collect_vec();
    let device = pen.coords.device().clone();

    let time = std::time::Instant::now();
    let mut fxc: Tsr = rt::zeros((dm1.shape().to_vec(), &device));
    for start in (0..pen.ngrids).step_by(GRID_BATCH) {
        let stop = (start + GRID_BATCH).min(pen.ngrids);
        let coords = &coords[start..stop];
        let weights = &weights[start..stop];
        let mut ni_obj = NIMatmul::new(pen.cint(), coords, weights);

        // generate rho
        let rho0 = ni_obj.make_rho_from_homogeneous_braket(&[bra.view()], TAU).unwrap();
        let rho1 = ni_obj.make_rho_from_dm(&dm1_list, TAU).unwrap();
        let xc_func = LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSSH", Unpolarized);
        let [_, _, fxc_eff] = libxc_eval_eff(&xc_func, rho0.i((.., .., 0)), 2, true).unwrap().try_into().unwrap();
        let fxc_batch = ni_obj.make_fxc_pot_with_eff(fxc_eff.view(), rho1.view(), TAU, 0).unwrap();
        fxc += fxc_batch;
    }
    println!("fxc fp: {:?}", fp(fxc.view()));
    println!("Time taken: {:.3} seconds", time.elapsed().as_secs_f64());
    fp_assert_eq!(fxc.view(), -3.2229077357662654, 1e-4);
}
