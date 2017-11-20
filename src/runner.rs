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

/// The result of one iteration.
#[derive(Debug)]
pub struct Iteration<D>
where
    D: DimName,
{
    /// The moving points.
    pub moved: Matrix<D>,

    /// The sigma2 value calculted by the registration for this iteration.
    pub sigma2: f64,
}

/// The result of a cpd run.
#[derive(Clone, Copy, Debug)]
pub struct Run {
    /// Did this run converge?
    pub converged: bool,
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

    // TODO add setters for outlier weight, sigma2, others

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
        mut registration: R,
    ) -> Result<(R::Transform, Run), Error>
    where
        R: Registration<D> + Into<<R as Registration<D>>::Transform>,
        D: DimName,
        UInt: Mul<<D as DimName>::Value>,
        <D as DimName>::Value: Mul + Mul<UInt>,
        <UInt as Mul<<D as DimName>::Value>>::Output: ArrayLength<f64>,
        <<D as DimName>::Value as Mul>::Output: ArrayLength<f64>,
        <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
    {
        let (fixed, moving, normalization) = self.normalize.normalize(fixed, moving);
        let mut error = 0.;
        let mut error_change = f64::MAX;
        let mut iterations = 0;
        let mut iteration = Iteration {
            moved: moving.as_ref().clone(),
            sigma2: self.sigma2.unwrap_or(sigma2(&fixed, &moving)),
        };
        let transformer = Transformer::new(&fixed, self.outlier_weight)?;
        while iterations < self.max_iterations && self.error_change_threshold < error_change &&
            self.sigma2_threshold < iteration.sigma2
        {
            let probabilities = transformer.probabilities(&iteration.moved, iteration.sigma2);
            error_change = ((probabilities.error - error) / probabilities.error).abs();
            info!(
                "iterations={}, error_change={}, sigma2={}",
                iterations,
                error_change,
                iteration.sigma2
            );
            error = probabilities.error;
            iteration = registration.iterate(&fixed, &moving, &probabilities);
            iterations += 1;
        }
        if let Some(normalization) = normalization {
            registration.denormalize(&normalization);
        }
        let run = Run { converged: iterations < self.max_iterations };
        Ok((registration.into(), run))
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
