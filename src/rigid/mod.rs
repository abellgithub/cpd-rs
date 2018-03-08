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
//! let run = Rigid::new().register(&matrix, &matrix).unwrap();
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

/// Build and run rigid registrations.
///
/// # Examples
///
/// Build:
///
/// ```
/// use cpd::Rigid;
/// let rigid = Rigid::new().scale(true);
/// ```
///
/// And run:
///
/// ```
/// use cpd::{Rigid, utils};
/// let matrix = utils::random_matrix2(10);
/// let run = Rigid::new().register(&matrix, &matrix).unwrap();
/// ```
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

    /// Returns this rigid configuration as a registration.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::{Rigid, U2};
    /// let rigid = Rigid::new();
    /// let registration = rigid.as_registration::<U2>().unwrap();
    /// ```
    pub fn as_registration<D>(
        &self,
    ) -> Result<Registration<D>, CannotNormalizeIndependentlyWithoutScale>
    where
        D: DimName,
        <D as DimName>::Value: Mul + Mul<UInt>,
        <<D as DimName>::Value as Mul>::Output: ArrayLength<f64>,
        <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
    {
        Registration::new(self)
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
    /// let run = rigid.register(&fixed, &moving).unwrap();
    /// ```
    pub fn register<D>(
        &self,
        fixed: &Matrix<D>,
        moving: &Matrix<D>,
    ) -> Result<Run<D, Transform<D>>, Error>
    where
        D: DimName + DimMin<D> + DimMin<D, Output = D> + DimSub<U1>,
        UInt: Mul<<D as DimName>::Value>,
        <UInt as Mul<<D as DimName>::Value>>::Output: ArrayLength<f64>,
        <D as DimName>::Value: Mul + Mul<UInt>,
        <<D as DimName>::Value as Mul>::Output: ArrayLength<f64>,
        <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
        DefaultAllocator: Allocator<f64, D, D>
            + Allocator<(usize, usize), D>
            + Allocator<f64, <D as DimSub<U1>>::Output>,
    {
        let registration = self.as_registration()?;
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
    macro_rules! rigid {
        ($name:ident, $normalize:expr, $scale:expr) => {
            mod $name {
                use {Matrix, SquareMatrix, Vector};
                use rigid::Transform;
                use nalgebra::{U2, Rotation2};

                const SCALE: f64 = 2.0;

                fn fixed() -> Matrix<U2> {
                    ::utils::matrix2_from_slice(&[1., 1., 1., 2., 1., 2., 3., 1.])
                }

                fn moving() -> Matrix<U2> {
                    let mut moving = fixed();
                    if $scale {
                        moving *= 1. / SCALE;
                    }
                    moving
                }

                fn rigid(fixed: Matrix<U2>, moving: Matrix<U2>) -> Transform<U2> {
                    use {Runner, Normalize};

                    let rigid = Runner::new()
                        .normalize($normalize)
                        .rigid()
                        .scale($scale);
                    let run = rigid.register(&fixed, &moving).unwrap();
                    assert!(run.converged);
                    assert_relative_eq!(fixed, run.moved, epsilon = 1e-4);
                    if $scale {
                        assert_relative_eq!(SCALE, run.transform.scale.unwrap(), epsilon = 1e-8);
                    } else {
                        assert_eq!(None, run.transform.scale);
                    }
                    run.transform
                }

                #[test]
                fn identity() {
                    let fixed = fixed();
                    let moving = moving();
                    let transform = rigid(fixed, moving);
                    assert_relative_eq!(
                        SquareMatrix::<U2>::identity(),
                        transform.rotation,
                        epsilon = 1e-8
                        );
                    assert_relative_eq!(Vector::<U2>::zeros(), transform.translation, epsilon = 1e-8);
                }

                #[test]
                fn rotation() {
                    let rotation = Rotation2::new(0.5);
                    let fixed = fixed();
                    let moving = moving() * rotation;
                    let transform = rigid(fixed, moving);
                    assert_relative_eq!(
                        *rotation.matrix(),
                        transform.rotation,
                        epsilon = 1e-8
                        );
                    assert_relative_eq!(Vector::<U2>::zeros(), transform.translation, epsilon = 1e-8);
                }

                #[test]
                fn translation() {
                    let translation = Vector::<U2>::new(1., 2.);
                    let fixed = fixed();
                    let mut moving = moving();
                    for d in 0..2 {
                        moving.column_mut(d).add_scalar_mut(-translation[d] / if $scale { SCALE } else { 1. });
                    }
                    let transform = rigid(fixed, moving);
                    assert_relative_eq!(
                        SquareMatrix::<U2>::identity(),
                        transform.rotation,
                        epsilon = 1e-8
                        );
                    assert_relative_eq!(translation, transform.translation, epsilon = 1e-8);
                }
            }
        }
    }

    rigid!(independent_and_scale, Normalize::Independent, true);
    rigid!(same_scale_and_scale, Normalize::SameScale, true);
    rigid!(same_scale_no_scale, Normalize::SameScale, false);
    rigid!(none_and_scale, Normalize::None, true);
    rigid!(none_no_scale, Normalize::None, false);
}
