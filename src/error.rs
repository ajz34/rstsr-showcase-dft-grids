#[derive(Debug, Clone)]
pub enum NIError {
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize>, msg: String },
    Message(String),
}

impl std::fmt::Display for NIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // use debug printing for simplicity
        write!(f, "NIError: {self:#?}")
    }
}
impl std::error::Error for NIError {}

pub(crate) trait NIIntoUsizeVec {
    fn into_usize_vec(self) -> Vec<usize>;
}

impl NIIntoUsizeVec for usize {
    fn into_usize_vec(self) -> Vec<usize> {
        vec![self]
    }
}

impl NIIntoUsizeVec for &[usize] {
    fn into_usize_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> NIIntoUsizeVec for [usize; N] {
    fn into_usize_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl NIIntoUsizeVec for Vec<usize> {
    fn into_usize_vec(self) -> Vec<usize> {
        self
    }
}

#[macro_export]
macro_rules! ni_check_shape {
    ($expected:expr, $actual:expr, $msg:expr) => {{
        if $expected.into_usize_vec() != $actual.into_usize_vec() {
            let str_expected = stringify!($expected);
            let str_actual = stringify!($actual);
            Err(NIError::ShapeMismatch {
                expected: $expected.into_usize_vec(),
                actual: $actual.into_usize_vec(),
                msg: $msg.to_string() + &format!(" (expected: {str_expected}, actual: {str_actual})"),
            })
        } else {
            Ok(())
        }
    }};

    ($cond:expr, $msg:expr) => {{
        if !$cond {
            let str_cond = stringify!($cond);
            Err(NIError::ShapeMismatch {
                expected: vec![],
                actual: vec![],
                msg: $msg.to_string() + &format!(" (condition: {str_cond})"),
            })
        } else {
            Ok(())
        }
    }};
}

#[macro_export]
macro_rules! ni_error {
    ($msg:expr) => {
        NIError::Message($msg.to_string())
    };

    ($msg:expr, $($arg:tt)*) => {
        NIError::Message(format!($msg, $($arg)*))
    };
}
