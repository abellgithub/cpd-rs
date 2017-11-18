//! Calculate the Gauss transform between to datasets.

mod probabilities;
mod transformer;

pub use self::probabilities::Probabilities;
pub use self::transformer::Transformer;

/// An error returned if the outlier weight is not between zero and one.
#[derive(Clone, Copy, Debug, Fail, PartialEq)]
#[fail(display = "Outlier weight is not between zero and one: {}", _0)]
pub struct InvalidOutlierWeight(f64);
