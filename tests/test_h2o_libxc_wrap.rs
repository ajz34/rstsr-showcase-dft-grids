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
fn h2o() -> H2OMolecule {
    H2OMolecule::load()
}

// ---------------------------------------------------------------------------
// Serial evaluation
// ---------------------------------------------------------------------------

mod test_eval_xc_serial {
    use super::*;

    #[rstest]
    fn test_rho(h2o: &H2OMolecule) {
        let mut ni_obj = h2o.build_ni_obj();
        let rho_tau = ni_obj.make_rho_from_dm(&[h2o.rdm1.view()], TAU).unwrap().into_squeeze(-1);
        let rho_rho = rho_tau.i((.., ..1));
        let xc_func = LibXCFunctional::from_identifier("lda_x", Unpolarized);
        let xc_eff = libxc_eval_eff_serial(&xc_func, rho_rho.view(), 3).unwrap();
        let fps = [
            fp((&xc_eff[0] * &h2o.weights).view()),
            fp((&xc_eff[1] * &h2o.weights).view()),
            fp((&xc_eff[2] * &h2o.weights * &rho_rho).view()),
            fp((&xc_eff[3] * &h2o.weights * &rho_rho * &rho_rho.i((.., None, ..))).view()),
        ];
        assert!((fps[0] - -0.0653646142).abs() < 1e-6);
        assert!((fps[1] - -0.0871528189).abs() < 1e-6);
        assert!((fps[2] - -0.0290509396).abs() < 1e-6);
        assert!((fps[3] - 0.0193672931).abs() < 1e-6);
    }

    #[rstest]
    fn test_sigma(h2o: &H2OMolecule) {
        let mut ni_obj = h2o.build_ni_obj();
        let rho_tau = ni_obj.make_rho_from_dm(&[h2o.rdm1.view()], TAU).unwrap().into_squeeze(-1);
        let rho_sigma = rho_tau.i((.., ..4));
        let xc_func = LibXCFunctional::from_identifier("gga_x_pbe", Unpolarized);
        let xc_eff = libxc_eval_eff_serial(&xc_func, rho_sigma.view(), 3).unwrap();
        let fps = [
            fp((&xc_eff[0] * &h2o.weights).view()),
            fp((&xc_eff[1] * &h2o.weights).view()),
            fp((&xc_eff[2] * &h2o.weights * &rho_sigma).view()),
            fp((&xc_eff[3] * &h2o.weights * &rho_sigma * &rho_sigma.i((.., None, ..))).view()),
        ];
        assert!((fps[0] - -0.1652985961).abs() < 1e-6);
        assert!((fps[1] - -0.2296325511).abs() < 1e-6);
        assert!((fps[2] - -0.1386410873).abs() < 1e-6);
        assert!((fps[3] - -0.523551635).abs() < 1e-6);
    }

    #[rstest]
    fn test_tau(h2o: &H2OMolecule) {
        let mut ni_obj = h2o.build_ni_obj();
        let rho_tau = ni_obj.make_rho_from_dm(&[h2o.rdm1.view()], TAU).unwrap().into_squeeze(-1);
        let xc_func = LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSSH", Unpolarized);
        let xc_eff = libxc_eval_eff_serial(&xc_func, rho_tau.view(), 3).unwrap();
        let fps = [
            fp((&xc_eff[0] * &h2o.weights).view()),
            fp((&xc_eff[1] * &h2o.weights).view()),
            fp((&xc_eff[2] * &h2o.weights * &rho_tau).view()),
            fp((&xc_eff[3] * &h2o.weights * &rho_tau * &rho_tau.i((.., None, ..))).view()),
        ];
        assert!((fps[0] - -0.1378400498).abs() < 1e-6);
        assert!((fps[1] - -0.1893572125).abs() < 1e-6);
        assert!((fps[2] - -0.1183454954).abs() < 1e-6);
        assert!((fps[3] - -1.0447740367).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Parallel evaluation
// ---------------------------------------------------------------------------

mod test_eval_xc_parallel {
    use super::*;

    #[rstest]
    fn test_rho(h2o: &H2OMolecule) {
        let mut ni_obj = h2o.build_ni_obj();
        let rho_tau = ni_obj.make_rho_from_dm(&[h2o.rdm1.view()], TAU).unwrap().into_squeeze(-1);
        let rho_rho = rho_tau.i((.., ..1));
        let xc_func = LibXCFunctional::from_identifier("lda_x", Unpolarized);
        let xc_eff = libxc_eval_eff_parallel(&xc_func, rho_rho.view(), 3, None).unwrap();
        let fps = [
            fp((&xc_eff[0] * &h2o.weights).view()),
            fp((&xc_eff[1] * &h2o.weights).view()),
            fp((&xc_eff[2] * &h2o.weights * &rho_rho).view()),
            fp((&xc_eff[3] * &h2o.weights * &rho_rho * &rho_rho.i((.., None, ..))).view()),
        ];
        assert!((fps[0] - -0.0653646142).abs() < 1e-6);
        assert!((fps[1] - -0.0871528189).abs() < 1e-6);
        assert!((fps[2] - -0.0290509396).abs() < 1e-6);
        assert!((fps[3] - 0.0193672931).abs() < 1e-6);
    }

    #[rstest]
    fn test_sigma(h2o: &H2OMolecule) {
        let mut ni_obj = h2o.build_ni_obj();
        let rho_tau = ni_obj.make_rho_from_dm(&[h2o.rdm1.view()], TAU).unwrap().into_squeeze(-1);
        let rho_sigma = rho_tau.i((.., ..4));
        let xc_func = LibXCFunctional::from_identifier("gga_x_pbe", Unpolarized);
        let xc_eff = libxc_eval_eff_parallel(&xc_func, rho_sigma.view(), 3, None).unwrap();
        let fps = [
            fp((&xc_eff[0] * &h2o.weights).view()),
            fp((&xc_eff[1] * &h2o.weights).view()),
            fp((&xc_eff[2] * &h2o.weights * &rho_sigma).view()),
            fp((&xc_eff[3] * &h2o.weights * &rho_sigma * &rho_sigma.i((.., None, ..))).view()),
        ];
        assert!((fps[0] - -0.1652985961).abs() < 1e-6);
        assert!((fps[1] - -0.2296325511).abs() < 1e-6);
        assert!((fps[2] - -0.1386410873).abs() < 1e-6);
        assert!((fps[3] - -0.523551635).abs() < 1e-6);
    }

    #[rstest]
    fn test_tau(h2o: &H2OMolecule) {
        let mut ni_obj = h2o.build_ni_obj();
        let rho_tau = ni_obj.make_rho_from_dm(&[h2o.rdm1.view()], TAU).unwrap().into_squeeze(-1);
        let xc_func = LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSSH", Unpolarized);
        let xc_eff = libxc_eval_eff_parallel(&xc_func, rho_tau.view(), 3, None).unwrap();
        let fps = [
            fp((&xc_eff[0] * &h2o.weights).view()),
            fp((&xc_eff[1] * &h2o.weights).view()),
            fp((&xc_eff[2] * &h2o.weights * &rho_tau).view()),
            fp((&xc_eff[3] * &h2o.weights * &rho_tau * &rho_tau.i((.., None, ..))).view()),
        ];
        assert!((fps[0] - -0.1378400498).abs() < 1e-6);
        assert!((fps[1] - -0.1893572125).abs() < 1e-6);
        assert!((fps[2] - -0.1183454954).abs() < 1e-6);
        assert!((fps[3] - -1.0447740367).abs() < 1e-6);
    }
}
