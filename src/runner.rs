use {Matrix, Normalize, Registration, Rigid, Run, UInt};
use generic_array::ArrayLength;
use nalgebra::DimName;
use std::ops::Mul;

/// Generic interface for running cpd registration methods.
///
/// Use the builder pattern to configure how cpd registrations are run.
///
/// ```
/// use cpd::Runner;
/// let runner = Runner::new().max_iterations(100); // etc
/// ```
///
/// Use methods like `rigid()` to specifiy the type of registration, and convert the builder into a
/// method-specific builder:
///
/// ```
/// use cpd::Runner;
/// let runner = Runner::new().rigid();
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Runner {
    max_iterations: usize,
    normalize: Normalize,
}

impl Runner {
    /// Creates a new, default runner.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::Runner;
    /// let runner = Runner::new();
    /// ```
    pub fn new() -> Runner {
        Runner::default()
    }

    /// Sets the maximum number of iterations when running cpd.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::Runner;
    /// let runner = Runner::new().max_iterations(100);
    /// ```
    pub fn max_iterations(mut self, max_iterations: usize) -> Runner {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the normalization strategy.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::{Normalize, Runner};
    /// let runner = Runner::new().normalize(Normalize::Independent);
    /// ```
    pub fn normalize(mut self, normalize: Normalize) -> Runner {
        self.normalize = normalize;
        self
    }

    /// Returns true if this runner requires scaling, usually because of the normalization.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::{Normalize, Runner};
    /// assert!(Runner::new().normalize(Normalize::Independent).requires_scaling());
    /// assert!(!Runner::new().normalize(Normalize::SameScale).requires_scaling());
    /// ```
    pub fn requires_scaling(&self) -> bool {
        self.normalize.requires_scaling()
    }

    /// Returns a rigid registration builder that will use this runner.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::Runner;
    /// let rigid = Runner::new().rigid();
    /// ```
    pub fn rigid(self) -> Rigid {
        self.into()
    }

    /// Runs a `Registration`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::{Runner, Rigid, utils, U2};
    ///
    /// let runner = Runner::new();
    /// let rigid = Rigid::new();
    /// let registration = rigid.as_registration::<U2>().unwrap();
    /// let matrix = utils::random_matrix2(10);
    /// let (transform, run) = runner.run(&matrix, &matrix, registration);
    /// ```
    pub fn run<D, R>(
        &self,
        fixed: &Matrix<D>,
        moving: &Matrix<D>,
        _registration: R,
    ) -> (R::Transform, Run)
    where
        R: Registration,
        D: DimName,
        <D as DimName>::Value: Mul + Mul<UInt>,
        <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
    {
        let (_fixed, _moving, _normalization) = self.normalize.normalize(fixed, moving);
        unimplemented!()
    }
}
