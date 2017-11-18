mod transform;

pub use self::transform::Transform;

use {Matrix, Run, Runner, UInt};
use generic_array::ArrayLength;
use nalgebra::DimName;
use std::ops::Mul;

/// An error that is returned when asked to normalize independenty without scaling.
#[derive(Clone, Copy, Debug, Fail)]
#[fail(display = "Cannot use Normalize::Independent without rigid scaling")]
pub struct CannotNormalizeIndependentlyWithoutScale {}

/// Builds rigid registrations.
///
/// Rigid registrations calculate rotations, translations, and optionally scaling to align to point
/// sets. To enable scaling, use the rigid builder:
///
/// ```
/// use cpd::Rigid;
/// let rigid = Rigid::new().scale(true);
/// ```
///
/// The rigid algorithm can check for reflections and prevent them from happening, which is the
/// default behavior. You must explicitly allow reflections:
///
/// ```
/// use cpd::Rigid;
/// let rigid = Rigid::new().allow_reflections(true);
/// ```
///
/// Use `register` to register two points sets:
///
/// ```
/// use cpd::{Rigid, utils};
/// let matrix = utils::random_matrix2(10);
/// let (transform, run) = Rigid::new().register(&matrix, &matrix).unwrap();
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
        _fixed: &Matrix<D>,
        _moving: &Matrix<D>,
    ) -> Result<(Transform<D>, Run), CannotNormalizeIndependentlyWithoutScale>
    where
        D: DimName,
        <D as DimName>::Value: Mul + Mul<UInt>,
        <<D as DimName>::Value as Mul>::Output: ArrayLength<f64>,
        <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
    {
        unimplemented!()
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
