//! Apply and de-apply scales and offsets to matrices.

use {Matrix, UInt, Vector};
use generic_array::ArrayLength;
use nalgebra::DimName;
use std::borrow::Cow;
use std::ops::Mul;

/// Normalization strategies.
///
/// By normalizing the points before running registrations, error sums can be kept smaller and, in
/// some cases, prevented from overflowing.
///
/// The default normalization strategy is SameScale:
///
/// ```
/// use cpd::Normalize;
/// assert_eq!(Normalize::SameScale, Normalize::default());
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Normalize {
    /// Normalize both point sets independently.
    ///
    /// When used with a rigid registration, requires that scaling be enabled.
    Independent,

    /// Normalize both points sets with the same scale value.
    ///
    /// Useful for LiDAR data, where you might want to reduce the coordinate values but you don't
    /// want to scale the points.
    SameScale,

    /// Don't normalize the points.
    None,
}

/// Parameters for the fixed and moving matrices that were used to normalize, and can be used to
/// denormalize.
#[derive(Debug, PartialEq)]
pub struct Normalization<D>
where
    D: DimName,
    <D as DimName>::Value: Mul + Mul<UInt>,
    <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
{
    /// The normalization parameters for the fixed points.
    pub fixed: Parameters<D>,

    /// The normzliation parameters for the moving points.
    pub moving: Parameters<D>,
}

/// Normalization parameters.
#[derive(Debug, PartialEq)]
pub struct Parameters<D>
where
    D: DimName,
    <D as DimName>::Value: Mul + Mul<UInt>,
    <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
{
    /// The offset of the points.
    pub offset: Vector<D>,

    /// The scaling applied to the points.
    pub scale: f64,
}

impl Normalize {
    /// Returns true if this normalization requires scaling.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::Normalize;
    /// assert!(Normalize::Independent.requires_scaling());
    /// assert!(!Normalize::SameScale.requires_scaling());
    /// assert!(!Normalize::None.requires_scaling());
    /// ```
    pub fn requires_scaling(&self) -> bool {
        match *self {
            Normalize::Independent => true,
            _ => false,
        }
    }

    /// Normalizes two matrices.
    ///
    /// Returns the normalized matrices (as `Cow`s, since they may or may not be modified) and an
    /// optional normalization structure.
    ///
    /// # Examples
    ///
    /// `Normalize::None` doesn't do anything:
    ///
    /// ```
    /// use cpd::{Normalize, utils};
    /// let matrix = utils::random_matrix2(10);
    /// let (fixed, moving, normalization) = Normalize::None.normalize(&matrix, &matrix);
    /// assert_eq!(matrix, *fixed);
    /// assert_eq!(matrix, *moving);
    /// assert_eq!(None, normalization);
    /// ```
    ///
    /// The other two change the matrices and return a structure that can be used to back-convert:
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate approx;
    /// # extern crate cpd;
    /// # fn main() {
    /// use cpd::{Normalize, utils};
    /// let fixed = utils::random_matrix2(10);
    /// let moving = utils::random_matrix2(10);
    /// let (mut fixed2, mut moving2, normalization) = Normalize::Independent.normalize(&fixed, &moving);
    /// let mut fixed2 = fixed2.to_mut();
    /// let mut moving2 = moving2.to_mut();
    /// let normalization = normalization.unwrap();
    /// normalization.fixed.denormalize(&mut fixed2);
    /// assert_relative_eq!(fixed, *fixed2);
    /// normalization.moving.denormalize(&mut moving2);
    /// assert_relative_eq!(moving, *moving2);
    /// # }
    /// ```
    pub fn normalize<'a, D>(
        &self,
        fixed: &'a Matrix<D>,
        moving: &'a Matrix<D>,
    ) -> (Cow<'a, Matrix<D>>, Cow<'a, Matrix<D>>, Option<Normalization<D>>)
    where
        D: DimName,
        <D as DimName>::Value: Mul + Mul<UInt>,
        <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
    {
        match *self {
            Normalize::Independent => {
                let normalization = Normalization::new(fixed, moving);
                let mut fixed = fixed.clone();
                let mut moving = moving.clone();
                normalization.normalize(&mut fixed, &mut moving);
                (Cow::Owned(fixed), Cow::Owned(moving), Some(normalization))
            }
            Normalize::SameScale => {
                let mut normalization = Normalization::new(fixed, moving);
                normalization.set_scales_to_mean();
                let mut fixed = fixed.clone();
                let mut moving = moving.clone();
                normalization.normalize(&mut fixed, &mut moving);
                (Cow::Owned(fixed), Cow::Owned(moving), Some(normalization))
            }
            Normalize::None => (Cow::Borrowed(fixed), Cow::Borrowed(moving), None),
        }
    }
}

impl Default for Normalize {
    fn default() -> Normalize {
        Normalize::SameScale
    }
}

impl<D> Normalization<D>
where
    D: DimName,
    <D as DimName>::Value: Mul + Mul<UInt>,
    <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
{
    fn new(fixed: &Matrix<D>, moving: &Matrix<D>) -> Normalization<D> {
        Normalization {
            fixed: Parameters::new(fixed),
            moving: Parameters::new(moving),
        }
    }

    fn set_scales_to_mean(&mut self) {
        let scale = (self.fixed.scale + self.moving.scale) / 2.;
        self.fixed.scale = scale;
        self.moving.scale = scale;
    }

    fn normalize(&self, fixed: &mut Matrix<D>, moving: &mut Matrix<D>) {
        self.fixed.normalize(fixed);
        self.moving.normalize(moving);
    }
}

impl<D> Default for Normalization<D>
where
    D: DimName,
    <D as DimName>::Value: Mul + Mul<UInt>,
    <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
{
    fn default() -> Normalization<D> {
        Normalization {
            fixed: Parameters::default(),
            moving: Parameters::default(),
        }
    }
}

impl<D> Parameters<D>
where
    D: DimName,
    <D as DimName>::Value: Mul + Mul<UInt>,
    <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
{
    /// Creates new parameters from a matrix.
    ///
    /// # Examples
    ///
    ///
    /// ```
    /// use cpd::{utils};
    /// use cpd::normalize::Parameters;
    /// let matrix = utils::random_matrix2(10);
    /// let parameters = Parameters::new(&matrix);
    /// ```
    pub fn new(matrix: &Matrix<D>) -> Parameters<D> {
        let offset = Vector::<D>::from_iterator((0..D::dim()).map(|d| {
            matrix.column(d).iter().sum::<f64>() / matrix.nrows() as f64
        }));
        let mut matrix = matrix.clone();
        for d in 0..D::dim() {
            matrix.column_mut(d).add_scalar_mut(-offset[d]);
        }
        let scale = (matrix.iter().map(|n| n.powi(2)).sum::<f64>() / matrix.nrows() as f64).sqrt();
        Parameters {
            offset: offset,
            scale: scale,
        }
    }

    /// Normalizes a matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::{U2, utils};
    /// use cpd::normalize::Parameters;
    /// let parameters = Parameters::<U2>::default();
    /// let matrix = utils::random_matrix2(10);
    /// let mut matrix2 = matrix.clone();
    /// parameters.normalize(&mut matrix2);
    /// assert_eq!(matrix, matrix2);
    /// ```
    pub fn normalize(&self, matrix: &mut Matrix<D>) {
        for d in 0..D::dim() {
            matrix.column_mut(d).add_scalar_mut(-self.offset[d]);
        }
        *matrix /= self.scale;
    }

    /// Denormalizes a matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::{U2, utils};
    /// use cpd::normalize::Parameters;
    /// let parameters = Parameters::<U2>::default();
    /// let matrix = utils::random_matrix2(10);
    /// let mut matrix2 = matrix.clone();
    /// parameters.denormalize(&mut matrix2);
    /// assert_eq!(matrix, matrix2);
    /// ```
    pub fn denormalize(&self, matrix: &mut Matrix<D>) {
        *matrix *= self.scale;
        for d in 0..D::dim() {
            matrix.column_mut(d).add_scalar_mut(self.offset[d]);
        }
    }
}

impl<D> Default for Parameters<D>
where
    D: DimName,
    <D as DimName>::Value: Mul + Mul<UInt>,
    <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
{
    fn default() -> Parameters<D> {
        Parameters {
            offset: Vector::<D>::zeros(),
            scale: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use {U2, Vector, utils};

    #[test]
    fn parameters() {
        let matrix = utils::matrix2_from_slice(&[1., 3., 2., 4.]);
        let parameters = Parameters::new(&matrix);
        assert_eq!(Vector::<U2>::new(2., 3.), parameters.offset);
    }

    #[test]
    fn independent_different_scale() {
        let fixed = utils::random_matrix2(10);
        let moving = &fixed * 2.;
        let (fixed, moving, normalization) = Normalize::Independent.normalize(&fixed, &moving);
        assert_relative_eq!(*fixed, *moving);
        let normalization = normalization.unwrap();
        assert_relative_eq!(normalization.fixed.scale, normalization.moving.scale / 2.0);
    }

    #[test]
    fn same_scale() {
        let fixed = utils::random_matrix2(10);
        let moving = &fixed * 2.;
        let (fixed, moving, normalization) = Normalize::SameScale.normalize(&fixed, &moving);
        assert!(*fixed != *moving);
        let normalization = normalization.unwrap();
        assert_relative_eq!(normalization.fixed.scale, normalization.moving.scale);
    }
}
