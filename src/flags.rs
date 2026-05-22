#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NIDenType {
    RHO,
    SIGMA,
    TAU,
    LAPL,
}

impl NIDenType {
    /// Returns the number of components in the output density for this XC type.
    ///
    /// - RHO: 1 component (density)
    /// - SIGMA: 4 components (density + 3 gradient components)
    /// - TAU: 5 components (density + 3 gradient components + kinetic energy density)
    /// - LAPL: 6 components (density + 3 gradient components + kinetic energy density + laplacian)
    pub fn num_rho_components(&self) -> usize {
        match self {
            NIDenType::RHO => 1,
            NIDenType::SIGMA => 4,
            NIDenType::TAU => 5,
            NIDenType::LAPL => 6,
        }
    }

    /// Returns the required AO derivative level for this XC type.
    ///
    /// - RHO: 0th order
    /// - SIGMA: 1st order (gradient)
    /// - TAU: 1st order (gradient)
    /// - LAPL: 2nd order (Laplacian)
    pub fn num_ao_deriv(&self) -> usize {
        match self {
            NIDenType::RHO => 0,
            NIDenType::SIGMA => 1,
            NIDenType::TAU => 1,
            NIDenType::LAPL => 2,
        }
    }
}
