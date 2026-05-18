pub mod flags;
pub mod numint_matmul;
pub mod pure_numint_matmul;

pub mod prelude {
    #![allow(unused)]

    use super::*;
    pub use flags::*;
    pub use numint_matmul::*;
    pub use pure_numint_matmul::*;

    pub(crate) use core::assert_matches;
    pub(crate) use libcint::prelude::*;
    pub(crate) use rstsr::prelude::*;
    pub(crate) use std::collections::HashMap;

    pub(crate) type DeviceTsr = DeviceFaer;
    pub(crate) type Tsr = Tensor<f64, DeviceTsr, IxD>;
    pub(crate) type TsrView<'a> = TensorView<'a, f64, DeviceTsr, IxD>;
    pub(crate) type TsrMut<'a> = TensorMut<'a, f64, DeviceTsr, IxD>;
}
