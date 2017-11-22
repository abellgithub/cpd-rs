//! Run cpd algorithms.

use {Matrix, Normalize, Registration, Rigid, UInt};
use failure::Error;
use gauss_transform::Transformer;
use generic_array::ArrayLength;
use nalgebra::DimName;
use std::f64;
use std::ops::Mul;

const DEFAULT_ERROR_CHANGE_THRESHOLD: f64 = 1e-5;
const DEFAULT_MAX_ITERATIONS: usize = 150;
const DEFAULT_OUTLIER_WEIGHT: f64 = 0.1;
const DEFAULT_SIGMA2_THRESHOLD: f64 = f64::EPSILON * 10.;

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
#[derive(Clone, Copy, Debug)]
pub struct Runner {
    error_change_threshold: f64,
    max_iterations: usize,
    normalize: Normalize,
    outlier_weight: f64,
    sigma2: Option<f64>,
    sigma2_threshold: f64,
}

/// The result of a cpd run.
#[derive(Debug)]
pub struct Run<D, T>
where
    D: DimName,
{
    /// Did this run converge?
    pub converged: bool,

    /// The number of iterations.
    pub iterations: usize,

    /// The moved points.
    pub moved: Matrix<D>,

    /// The transform returned by the registration method.
    pub transform: T,
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

    /// Sets the error change threshold.
    ///
    /// Make this lower if you want to get more precise.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::Runner;
    /// let runner = Runner::new().error_change_threshold(1e-8);
    /// ```
    pub fn error_change_threshold(mut self, error_change_threshold: f64) -> Runner {
        self.error_change_threshold = error_change_threshold;
        self
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

    /// Sets the outlier weight.
    ///
    /// Does *not* check to see whether it is a valid value, yet.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::Runner;
    /// let runner = Runner::new().outlier_weight(0.2);
    /// ```
    pub fn outlier_weight(mut self, outlier_weight: f64) -> Runner {
        self.outlier_weight = outlier_weight;
        self
    }

    /// Sets the initial sigma2.
    ///
    /// If none, use the default sigma2 as calculated from the matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::Runner;
    /// let runner = Runner::new().sigma2(1.1).sigma2(None);
    /// ```
    pub fn sigma2<T: Into<Option<f64>>>(mut self, sigma2: T) -> Runner {
        self.sigma2 = sigma2.into();
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
    /// let run = runner.run(&matrix, &matrix, registration).unwrap();
    /// ```
    pub fn run<D, R>(
        &self,
        fixed: &Matrix<D>,
        moving: &Matrix<D>,
        mut registration: R,
    ) -> Result<Run<D, R::Transform>, Error>
    where
        R: Registration<D> + Into<<R as Registration<D>>::Transform>,
        D: DimName,
        UInt: Mul<<D as DimName>::Value>,
        <D as DimName>::Value: Mul + Mul<UInt>,
        <UInt as Mul<<D as DimName>::Value>>::Output: ArrayLength<f64>,
        <<D as DimName>::Value as Mul>::Output: ArrayLength<f64>,
        <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
    {
        let (fixed, mut moving, normalization) = self.normalize.normalize(fixed, moving);
        let mut error = 0.;
        let mut error_change = f64::MAX;
        let mut iterations = 0;
        let mut sigma2 = self.sigma2.unwrap_or(sigma2(&fixed, &moving));
        let mut moved = moving.as_ref().clone();
        let transformer = Transformer::new(&fixed, self.outlier_weight)?;
        while iterations < self.max_iterations && self.error_change_threshold < error_change &&
            self.sigma2_threshold < sigma2
        {
            let probabilities = transformer.probabilities(&moved, sigma2);
            error_change = ((probabilities.error - error) / probabilities.error).abs();
            info!(
                "iterations={}, error_change={}, sigma2={}",
                iterations,
                error_change,
                sigma2
            );
            error = probabilities.error;
            sigma2 = registration.iterate(&fixed, &moving, &probabilities);
            moved = registration.transform(&moving);
            iterations += 1;
        }
        if let Some(normalization) = normalization {
            registration.denormalize(&normalization);
            normalization.moving.denormalize(moving.to_mut());
        }
        moved = registration.transform(&moving);
        Ok(Run {
            converged: iterations < self.max_iterations,
            iterations: iterations,
            moved: moved,
            transform: registration.into(),
        })
    }
}

impl Default for Runner {
    fn default() -> Runner {
        Runner {
            error_change_threshold: DEFAULT_ERROR_CHANGE_THRESHOLD,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            normalize: Normalize::default(),
            outlier_weight: DEFAULT_OUTLIER_WEIGHT,
            sigma2: None,
            sigma2_threshold: DEFAULT_SIGMA2_THRESHOLD,
        }
    }
}

/// The default sigma2 for two matrices.
///
/// # Examples
///
/// ```
/// use cpd::{runner, utils};
/// let matrix = utils::random_matrix2(10);
/// let sigma2 = runner::sigma2(&matrix, &matrix);
/// ```
pub fn sigma2<D>(fixed: &Matrix<D>, moving: &Matrix<D>) -> f64
where
    D: DimName,
    UInt: Mul<<D as DimName>::Value>,
    <D as DimName>::Value: Mul + Mul<UInt>,
    <UInt as Mul<<D as DimName>::Value>>::Output: ArrayLength<f64>,
    <<D as DimName>::Value as Mul>::Output: ArrayLength<f64>,
    <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
{
    use RowVector;
    let sum = |matrix: &Matrix<D>| {
        RowVector::<D>::from_iterator((0..D::dim()).map(|d| matrix.column(d).iter().sum::<f64>()))
    };
    let numerator = fixed.nrows() as f64 * (fixed.transpose() * fixed).trace() +
        moving.nrows() as f64 * (moving.transpose() * moving).trace() -
        2. * (sum(fixed) * sum(moving).transpose())[0];
    let denomintaor = (fixed.nrows() * moving.nrows() * D::dim()) as f64;
    numerator / denomintaor
}

#[cfg(test)]
mod tests {
    use utils;

    #[test]
    fn sigma2() {
        let matrix = utils::matrix2_from_slice(&[1., 2., 3., 4.]);
        assert_relative_eq!(0.5, super::sigma2(&matrix, &matrix));
    }
}
