//! Basic utility functions for creating matrices.

use Matrix;
use nalgebra::U2;

/// Creates a random matrix with two columns and configurable rows.
///
/// # Examples
///
/// ```
/// use cpd::utils;
/// let matrix = utils::random_matrix2(10);
/// assert_eq!(2, matrix.ncols());
/// assert_eq!(10, matrix.nrows());
/// ```
pub fn random_matrix2(nrows: usize) -> Matrix<U2> {
    use nalgebra::U2;
    Matrix::<U2>::new_random(nrows)
}
