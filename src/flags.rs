#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NIXCType {
    LDA,
    GGA,
    MGGA,
    LAPL,
}

impl NIXCType {
    pub fn num_rho_components(&self) -> usize {
        match self {
            NIXCType::LDA => 1,
            NIXCType::GGA => 4,
            NIXCType::MGGA => 5,
            NIXCType::LAPL => 6,
        }
    }
}
