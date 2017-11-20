//! Calculate the [Gauss transform](https://www.encyclopediaofmath.org/index.php/Gauss_transform) between to datasets.
//!
//! Calculating a transform is broken into two steps. First, create a `Transformer`, providing the
//! fixed matrix and the outlier weight:
//!
//! ```
//! let fixed = cpd::utils::random_matrix2(10);
//! let transformer = cpd::gauss_transform::Transformer::new(&fixed, 0.1).unwrap();
//! ```
//!
//! The outlier weight is a number between zero and one that represents how much weight is given to
//! outlier points. Unless you have a compelling reason to not use 0.1, stick with that.
//!
//! Then, transform the points, providing the moving matrix and a sigma2 value:
//!
//! ```
//! let fixed = cpd::utils::random_matrix2(10);
//! let transformer = cpd::gauss_transform::Transformer::new(&fixed, 0.1).unwrap();
//! let moving = cpd::utils::random_matrix2(10);
//! let probabilities = transformer.probabilities(&moving, 1.0);
//! ```
//!
//! The sigma2 parameter defines the "spread" of the transform.
//!
//! For now, all transforms are done using the direct method, calculating the actual transform.
//! There are transform approximation algorithms out there that may be implemented in the future.

mod probabilities;
mod transformer;

pub use self::probabilities::Probabilities;
pub use self::transformer::Transformer;

/// An error returned if the outlier weight is not between zero and one.
#[derive(Clone, Copy, Debug, Fail, PartialEq)]
#[fail(display = "Outlier weight is not between zero and one: {}", _0)]
pub struct InvalidOutlierWeight(f64);
