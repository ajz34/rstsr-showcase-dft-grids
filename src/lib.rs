pub mod error;
pub mod flags;
pub mod numint_matmul;

pub mod prelude {
    #![allow(unused)]

    use super::*;
    pub use crate::ni_check_shape;
    pub use error::*;
    pub use flags::*;

    pub(crate) use core::assert_matches;
    pub(crate) use libcint::prelude::*;
    pub(crate) use rstsr::prelude::*;
    pub(crate) use std::collections::HashMap;

    pub(crate) type DeviceTsr = DeviceFaer;
    pub(crate) type Tsr = Tensor<f64, DeviceTsr, IxD>;
    pub(crate) type TsrView<'a> = TensorView<'a, f64, DeviceTsr, IxD>;
    pub(crate) type TsrMut<'a> = TensorMut<'a, f64, DeviceTsr, IxD>;
}
