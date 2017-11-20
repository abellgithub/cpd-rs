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

/// Creates a 2-column matrix from a slice, column-major.
///
/// # Examples
///
/// ```
/// use cpd::utils;
/// let matrix = utils::matrix2_from_slice(&[1., 3., 2., 4.]);
/// assert_eq!(2, matrix.nrows());
/// assert_eq!(2, matrix.ncols());
/// assert_eq!(3., matrix[(1, 0)]);
/// assert_eq!(2., matrix[(0, 1)]);
/// ```
pub fn matrix2_from_slice(slice: &[f64]) -> Matrix<U2> {
    Matrix::<U2>::from_iterator(slice.len() / 2, slice.iter().map(|&n| n))
}

#[cfg(feature = "las")]
/// Read las data into nalgebra matrices.
///
/// # Examples
///
/// ```
/// use cpd::utils;
/// let face = utils::matrix_from_las_path("tests/data/face.las").unwrap();
/// assert_eq!(392, face.nrows());
/// assert_eq!(3, face.ncols());
/// ```
pub fn matrix_from_las_path<P>(path: P) -> Result<Matrix<::nalgebra::U3>, ::failure::Error>
where
    P: AsRef<::std::path::Path>,
{
    use las::Reader;
    let mut reader = Reader::from_path(path)?;
    let mut matrix = Matrix::<::nalgebra::U3>::zeros(reader.header().number_of_points() as usize);
    for (i, point) in reader.points().enumerate() {
        let point = point?;
        matrix[(i, 0)] = point.x;
        matrix[(i, 1)] = point.y;
        matrix[(i, 2)] = point.z;
    }
    Ok(matrix)
}
