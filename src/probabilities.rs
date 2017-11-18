use Matrix;
use nalgebra::{DVector, DimName};

/// An error returned if the outlier weight is not between zero and one.
#[derive(Clone, Copy, Debug, Fail, PartialEq)]
#[fail(display = "Outlier weight is not between zero and one: {}", _0)]
pub struct InvalidOutlierWeight(f64);

/// The alignment probabilities between two datasets.
#[derive(Debug)]
pub struct Probabilities<D>
where
    D: DimName,
{
    /// A probability vector with the same length as the moving points.
    pub p1: DVector<f64>,

    /// A probability vector with the same length as the fixed points.
    pub pt1: DVector<f64>,

    /// A probability matrix with the same lenth as the moving points.
    pub px: Matrix<D>,

    /// The error between the two matrices.
    pub error: f64,
}

impl<D> Probabilities<D>
where
    D: DimName,
{
    /// Creates a new set of probabilities for two matrices, a sigma2, and an outlier weight.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::{Probabilities, utils};
    /// let fixed = utils::random_matrix2(10);
    /// let moving = utils::random_matrix2(10);
    /// let probabilities = Probabilities::new(&fixed, &moving, 1.0, 0.1);
    /// ```
    pub fn new(
        _fixed: &Matrix<D>,
        _moving: &Matrix<D>,
        _sigma2: f64,
        _outlier_weight: f64,
    ) -> Result<Probabilities<D>, InvalidOutlierWeight> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use utils;

    #[test]
    fn invalid_outlier_weight() {
        let matrix = utils::random_matrix2(10);
        assert_eq!(
            InvalidOutlierWeight(-1.),
            Probabilities::new(&matrix, &matrix, 1.0, -1.).unwrap_err()
        );
        assert_eq!(
            InvalidOutlierWeight(1.1),
            Probabilities::new(&matrix, &matrix, 1.0, 1.1).unwrap_err()
        );
    }
}
