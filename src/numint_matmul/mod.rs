//! Matrix multiplication driver for DFT numerical integration.
#![doc = include_str!("docs/mod.md")]

pub mod impls;
pub mod pure_eval_rho;
pub mod structs;

#[allow(unused)]
pub mod prelude {
    pub use crate::prelude::*;

    pub(crate) use super::impls::*;
    pub(crate) use super::pure_eval_rho::*;
    pub(crate) use super::structs::*;
}

#[allow(unused)]
use prelude::*;
