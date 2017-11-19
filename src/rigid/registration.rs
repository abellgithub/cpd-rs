use super::Rigid;
use {Iteration, Matrix, SquareMatrix, UInt, Vector};
use gauss_transform::Probabilities;
use generic_array::ArrayLength;
use nalgebra::{DefaultAllocator, DimMin, DimName, DimSub, U1};
use nalgebra::allocator::Allocator;
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
    type Transform = ::rigid::Transform<D>;

    fn iterate(&mut self, fixed: &Matrix<D>, moving: &Matrix<D>, probabilities: &Probabilities<D>) -> Iteration<D> {
        let np = probabilities.pt1.iter().sum::<f64>();
        let mu_fixed = fixed.transpose() * &probabilities.pt1  / np;
        let mu_moving = moving.transpose() * &probabilities.p1  / np;
        let a = probabilities.px.transpose() * moving  - np * &mu_fixed * mu_moving.transpose();
        let svd = a.svd(true, true);
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
