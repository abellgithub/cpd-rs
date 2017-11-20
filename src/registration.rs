use {Iteration, Matrix, Normalization, UInt};
use gauss_transform::Probabilities;
use generic_array::ArrayLength;
use nalgebra::DimName;
use std::ops::Mul;

/// A trait for all structures that can be registered by a runner.
pub trait Registration<D>
where
    D: DimName,
    <D as DimName>::Value: Mul + Mul<UInt>,
    <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
{
    /// The struct that is returned after a registration. Holds the registration information, e.g.
    /// rotation and translation matrices.
    type Transform;

    /// Perform one iteration of the registration.
    ///
    /// # Examples
    ///
    /// Rigid's registration implements `Registration`:
    ///
    /// ```
    /// use cpd::{Rigid, utils, Registration};
    /// use cpd::gauss_transform::Transformer;
    /// let rigid = Rigid::new();
    /// let mut registration = rigid.as_registration().unwrap();
    /// let matrix = utils::random_matrix2(10);
    /// let transformer = Transformer::new(&matrix, 0.1).unwrap();
    /// let probabilities = transformer.probabilities(&matrix, 1.0);
    /// let iteration = registration.iterate(&matrix, &matrix, &probabilities);
    /// ```
    fn iterate(
        &mut self,
        fixed: &Matrix<D>,
        moving: &Matrix<D>,
        probabilities: &Probabilities<D>,
    ) -> Iteration<D>;

    /// Denormalize the registration.
    ///
    /// # Examples
    ///
    /// Rigid's registration implements 'Registration':
    ///
    /// ```
    /// use cpd::{U2, Rigid, Registration, Normalization};
    /// let rigid = Rigid::new();
    /// let mut registration = rigid.as_registration::<U2>().unwrap();
    /// let normalization = Normalization::default();
    /// registration.denormalize(&normalization);
    /// ```
    fn denormalize(&mut self, normalization: &Normalization<D>);
}
