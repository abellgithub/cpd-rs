use super::Rigid;
use {Iteration, Matrix, Probabilities, SquareMatrix, UInt, Vector};
use generic_array::ArrayLength;
use nalgebra::DimName;
use std::ops::Mul;

/// An error that is returned when asked to normalize independenty without scaling.
#[derive(Clone, Copy, Debug, Fail)]
#[fail(display = "Cannot use Normalize::Independent without rigid scaling")]
pub struct CannotNormalizeIndependentlyWithoutScale {}

/// A `Registration` for running rigid registrations.
#[derive(Debug)]
pub struct Registration<'a, D>
where
    D: DimName,
    <D as DimName>::Value: Mul + Mul<UInt>,
    <<D as DimName>::Value as Mul>::Output: ArrayLength<f64>,
    <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
{
    rigid: &'a Rigid,
    rotation: SquareMatrix<D>,
    scale: f64,
    translation: Vector<D>,
}

impl<'a, D> Registration<'a, D>
where
    D: DimName,
    <D as DimName>::Value: Mul + Mul<UInt>,
    <<D as DimName>::Value as Mul>::Output: ArrayLength<f64>,
    <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
{
    /// Creates an new registration from a rigid.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::U2;
    /// use cpd::rigid::{Rigid, Registration};
    /// let rigid = Rigid::new();
    /// let registration = Registration::<U2>::new(&rigid).unwrap();
    /// ```
    pub fn new(
        rigid: &'a Rigid,
    ) -> Result<Registration<'a, D>, CannotNormalizeIndependentlyWithoutScale> {
        if rigid.runner.requires_scaling() && !rigid.scale {
            Err(CannotNormalizeIndependentlyWithoutScale {})
        } else {
            Ok(Registration {
                rigid: rigid,
                rotation: SquareMatrix::<D>::identity(),
                scale: 1.0,
                translation: Vector::<D>::zeros(),
            })
        }
    }
}

impl<'a, D> ::Registration<D> for Registration<'a, D>
where
    D: DimName,
    <D as DimName>::Value: Mul + Mul<UInt>,
    <<D as DimName>::Value as Mul>::Output: ArrayLength<f64>,
    <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
{
    type Transform = ::rigid::Transform<D>;

    fn iterate(&mut self, _fixed: &Matrix<D>, _moving: &Matrix<D>, _probabilities: &Probabilities<D>) -> Iteration<D> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use {Normalize, Runner};
    use nalgebra::U2;

    #[test]
    fn normalize_independent_and_no_scale() {
        let rigid = Runner::new()
            .normalize(Normalize::Independent)
            .rigid()
            .scale(false);
        assert!(Registration::<U2>::new(&rigid).is_err());
    }
}
