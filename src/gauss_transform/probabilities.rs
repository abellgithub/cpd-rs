use Matrix;
use nalgebra::{DVector, DimName};

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
