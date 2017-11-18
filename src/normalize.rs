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
    /// Useful for LiDAR data, where you might want to reduce the coordiante values but you don't
    /// want to scale the points.
    SameScale,

    /// Don't normalize the points.
    None,
}

/// The parameters that can be used to de-normalize points.
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
    /// use cpd::{Normalize, utils};
    /// let fixed = utils::random_matrix2(10);
    /// let moving = utils::random_matrix2(10);
    /// let (mut fixed2, mut moving2, normalization) = Normalize::Independent.normalize(&fixed, &moving);
    /// let mut fixed2 = fixed2.to_mut();
    /// let mut moving2 = moving2.to_mut();
    /// let normalization = normalization.unwrap();
    /// normalization.fixed.denormalize(&mut fixed2);
    /// assert_eq!(fixed, *fixed2);
    /// normalization.moving.denormalize(&mut moving2);
    /// assert_eq!(moving, *moving2);
    /// ```
    pub fn normalize<'a, D>(
        &self,
        _fixed: &'a Matrix<D>,
        _moving: &'a Matrix<D>,
    ) -> (Cow<'a, Matrix<D>>, Cow<'a, Matrix<D>>, Option<Normalization<D>>)
    where
        D: DimName,
        <D as DimName>::Value: Mul + Mul<UInt>,
        <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
    {
        unimplemented!()
    }
}

impl Default for Normalize {
    fn default() -> Normalize {
        Normalize::SameScale
    }
}

impl<D> Parameters<D>
where
    D: DimName,
    <D as DimName>::Value: Mul + Mul<UInt>,
    <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
{
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
    pub fn denormalize(&self, _matrix: &mut Matrix<D>) {
        unimplemented!()
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
