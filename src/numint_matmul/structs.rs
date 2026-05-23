use super::prelude::*;

pub struct NIMatMul<'a> {
    pub cint: CInt,
    pub coords: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub cache_tensor: HashMap<String, TsrCow<'a, f64>>,
}
