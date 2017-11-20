//! Run cpd's rigid registration.
//!
//! Rigid registrations calculate rotations, translations, and optionally scaling to align to point
//! sets. To enable scaling, use the rigid builder:
//!
//! ```
//! use cpd::Rigid;
//! let rigid = Rigid::new().scale(true);
//! ```
//!
//! The rigid algorithm can check for reflections and prevent them from happening, which is the
//! default behavior. You must explicitly allow reflections:
//!
//! ```
//! use cpd::Rigid;
//! let rigid = Rigid::new().allow_reflections(true);
//! ```
//!
//! Use `register` to register two points sets:
//!
//! ```
//! use cpd::{Rigid, utils};
//! let matrix = utils::random_matrix2(10);
//! let (transform, run) = Rigid::new().register(&matrix, &matrix).unwrap();
//! ```

mod registration;
mod transform;

pub use self::registration::{CannotNormalizeIndependentlyWithoutScale, Registration};
pub use self::transform::Transform;

use {Matrix, Run, Runner, UInt};
use failure::Error;
use generic_array::ArrayLength;
use nalgebra::{DefaultAllocator, DimMin, DimName, DimSub, U1};
use nalgebra::allocator::Allocator;
use std::ops::Mul;

/// The builder for rigid registrations.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rigid {
    allow_reflections: bool,
    runner: Runner,
    scale: bool,
}

impl Rigid {
    /// Creates a new rigid registration builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::Rigid;
    /// let rigid = Rigid::new();
    /// ```
    pub fn new() -> Rigid {
        Rigid::default()
    }

    /// The rigid registration can prevent the rotation matrix from reflecting the points, or not.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::Rigid;
    /// let rigid = Rigid::new().allow_reflections(true);
    /// ```
    pub fn allow_reflections(mut self, allow_reflections: bool) -> Rigid {
        self.allow_reflections = allow_reflections;
        self
    }

    /// The rigid registration can chose to scale the points, or not.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::Rigid;
    /// let rigid = Rigid::new().scale(true);
    /// ```
    pub fn scale(mut self, scale: bool) -> Rigid {
        self.scale = scale;
        self
    }

    /// Registers two matrices, returning the transform and information about the run.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::{Rigid, utils};
    /// let fixed = utils::random_matrix2(10);
    /// let moving = utils::random_matrix2(10);
    /// let rigid = Rigid::new();
    /// let (transform, run) = rigid.register(&fixed, &moving).unwrap();
    /// ```
    pub fn register<D>(
        &self,
        fixed: &Matrix<D>,
        moving: &Matrix<D>,
    ) -> Result<(Transform<D>, Run), Error>
    where
        D: DimName + DimMin<D> + DimMin<D, Output = D> + DimSub<U1>,
        UInt: Mul<<D as DimName>::Value>,
        <UInt as Mul<<D as DimName>::Value>>::Output: ArrayLength<f64>,
        <D as DimName>::Value: Mul + Mul<UInt>,
        <<D as DimName>::Value as Mul>::Output: ArrayLength<f64>,
        <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
        DefaultAllocator: Allocator<f64, D, D> +
            Allocator<(usize, usize), D> +
            Allocator<f64, <D as DimSub<U1>>::Output>,
    {
        let registration = Registration::new(self, moving)?;
        let tuple = self.runner.run(fixed, moving, registration)?;
        Ok(tuple)
    }
}

impl From<Runner> for Rigid {
    fn from(runner: Runner) -> Rigid {
        Rigid {
            runner: runner,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use {SquareMatrix, Vector, utils};
    use nalgebra::U2;

    // TODO test for each normalization
    #[test]
    fn identity() {
        let rigid = Rigid::new();
        let matrix = utils::matrix2_from_slice(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        let (transform, run) = rigid.register(&matrix, &matrix).unwrap();
        assert!(run.converged);
        assert_relative_eq!(SquareMatrix::<U2>::identity(), transform.rotation);
        assert_relative_eq!(Vector::<U2>::zeros(), transform.translation);
    }
}
