mod test_util;

use libxc::prelude::*;
use rstest::*;
use rstsr_showcase_dft_grids::prelude::*;
use test_util::*;

use LibXCSpin::*;
use NIDenType::*;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

#[fixture]
#[once]
fn ch2() -> Ch2Molecule {
    Ch2Molecule::load()
}

// ---------------------------------------------------------------------------
// Serial evaluation
// ---------------------------------------------------------------------------

mod test_eval_xc_serial {
    use super::*;

    #[rstest]
    fn test_rho(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let rho_tau = ni_obj.make_rho_from_dm(&[ch2.rdm1.i((.., .., 0)), ch2.rdm1.i((.., .., 1))], TAU).unwrap();
        let rho_rho = rho_tau.i((.., ..1, ..));
        let xc_func = LibXCFunctional::from_identifier("lda_x", Polarized);
        let xc_eff = libxc_eval_eff_serial(&xc_func, rho_rho.view(), 3).unwrap();
        let fps = [
            fp((&xc_eff[0] * &ch2.weights).view()),
            fp((&xc_eff[1] * &ch2.weights).view()),
            fp((&xc_eff[2] * &ch2.weights * &rho_rho).view()),
            fp((&xc_eff[3] * &ch2.weights * &rho_rho * &rho_rho.i((.., None, None, .., ..))).view()),
        ];
        assert!((fps[0] - -0.0050679843).abs() < 1e-6);
        assert!((fps[1] - 0.1013037713).abs() < 1e-6);
        assert!((fps[2] - -0.0417593614).abs() < 1e-6);
        assert!((fps[3] - 0.0281257118).abs() < 1e-6);
    }

    #[rstest]
    fn test_sigma(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let rho_tau = ni_obj.make_rho_from_dm(&[ch2.rdm1.i((.., .., 0)), ch2.rdm1.i((.., .., 1))], TAU).unwrap();
        let rho_sigma = rho_tau.i((.., ..4, ..));
        let xc_func = LibXCFunctional::from_identifier("gga_x_pbe", Polarized);
        let xc_eff = libxc_eval_eff_serial(&xc_func, rho_sigma.view(), 3).unwrap();
        let fps = [
            fp((&xc_eff[0] * &ch2.weights).view()),
            fp((&xc_eff[1] * &ch2.weights).view()),
            fp((&xc_eff[2] * &ch2.weights * &rho_sigma).view()),
            fp((&xc_eff[3] * &ch2.weights * &rho_sigma * &rho_sigma.i((.., None, None, .., ..))).view()),
        ];
        assert!((fps[0] - 0.0174826167).abs() < 1e-6);
        assert!((fps[1] - -0.0688243866).abs() < 1e-6);
        assert!((fps[2] - -0.0998561381).abs() < 1e-6);
        assert!((fps[3] - 0.1192110757).abs() < 1e-6);
    }

    #[rstest]
    fn test_tau(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let rho_tau = ni_obj.make_rho_from_dm(&[ch2.rdm1.i((.., .., 0)), ch2.rdm1.i((.., .., 1))], TAU).unwrap();
        let xc_func = LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSSH", Polarized);
        let xc_eff = libxc_eval_eff_serial(&xc_func, rho_tau.view(), 3).unwrap();
        let fps = [
            fp((&xc_eff[0] * &ch2.weights).view()),
            fp((&xc_eff[1] * &ch2.weights).view()),
            fp((&xc_eff[2] * &ch2.weights * &rho_tau).view()),
            fp((&xc_eff[3] * &ch2.weights * &rho_tau * &rho_tau.i((.., None, None, .., ..))).view()),
        ];
        assert!((fps[0] - 0.0070440179).abs() < 1e-6);
        assert!((fps[1] - 0.0931735851).abs() < 1e-6);
        assert!((fps[2] - -0.0147352726).abs() < 1e-6);
        assert!((fps[3] - 1.3842052458).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Parallel evaluation
// ---------------------------------------------------------------------------

mod test_eval_xc_parallel {
    use super::*;

    #[rstest]
    fn test_rho(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let rho_tau = ni_obj.make_rho_from_dm(&[ch2.rdm1.i((.., .., 0)), ch2.rdm1.i((.., .., 1))], TAU).unwrap();
        let rho_rho = rho_tau.i((.., ..1, ..));
        let xc_func = LibXCFunctional::from_identifier("lda_x", Polarized);
        let xc_eff = libxc_eval_eff_parallel(&xc_func, rho_rho.view(), 3, None).unwrap();
        let fps = [
            fp((&xc_eff[0] * &ch2.weights).view()),
            fp((&xc_eff[1] * &ch2.weights).view()),
            fp((&xc_eff[2] * &ch2.weights * &rho_rho).view()),
            fp((&xc_eff[3] * &ch2.weights * &rho_rho * &rho_rho.i((.., None, None, .., ..))).view()),
        ];
        assert!((fps[0] - -0.0050679843).abs() < 1e-6);
        assert!((fps[1] - 0.1013037713).abs() < 1e-6);
        assert!((fps[2] - -0.0417593614).abs() < 1e-6);
        assert!((fps[3] - 0.0281257118).abs() < 1e-6);
    }

    #[rstest]
    fn test_sigma(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let rho_tau = ni_obj.make_rho_from_dm(&[ch2.rdm1.i((.., .., 0)), ch2.rdm1.i((.., .., 1))], TAU).unwrap();
        let rho_sigma = rho_tau.i((.., ..4, ..));
        let xc_func = LibXCFunctional::from_identifier("gga_x_pbe", Polarized);
        let xc_eff = libxc_eval_eff_parallel(&xc_func, rho_sigma.view(), 3, None).unwrap();
        let fps = [
            fp((&xc_eff[0] * &ch2.weights).view()),
            fp((&xc_eff[1] * &ch2.weights).view()),
            fp((&xc_eff[2] * &ch2.weights * &rho_sigma).view()),
            fp((&xc_eff[3] * &ch2.weights * &rho_sigma * &rho_sigma.i((.., None, None, .., ..))).view()),
        ];
        assert!((fps[0] - 0.0174826167).abs() < 1e-6);
        assert!((fps[1] - -0.0688243866).abs() < 1e-6);
        assert!((fps[2] - -0.0998561381).abs() < 1e-6);
        assert!((fps[3] - 0.1192110757).abs() < 1e-6);
    }

    #[rstest]
    fn test_tau(ch2: &Ch2Molecule) {
        let mut ni_obj = ch2.build_ni_obj();
        let rho_tau = ni_obj.make_rho_from_dm(&[ch2.rdm1.i((.., .., 0)), ch2.rdm1.i((.., .., 1))], TAU).unwrap();
        let xc_func = LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSSH", Polarized);
        let xc_eff = libxc_eval_eff_parallel(&xc_func, rho_tau.view(), 3, None).unwrap();
        let fps = [
            fp((&xc_eff[0] * &ch2.weights).view()),
            fp((&xc_eff[1] * &ch2.weights).view()),
            fp((&xc_eff[2] * &ch2.weights * &rho_tau).view()),
            fp((&xc_eff[3] * &ch2.weights * &rho_tau * &rho_tau.i((.., None, None, .., ..))).view()),
        ];
        assert!((fps[0] - 0.0070440179).abs() < 1e-6);
        assert!((fps[1] - 0.0931735851).abs() < 1e-6);
        assert!((fps[2] - -0.0147352726).abs() < 1e-6);
        assert!((fps[3] - 1.3842052458).abs() < 1e-6);
    }
}
