use {Matrix, Normalization, Rigid, SquareMatrix, UInt, Vector};
use gauss_transform::Probabilities;
use generic_array::ArrayLength;
use nalgebra::{DefaultAllocator, DimMin, DimName, DimSub, U1};
use nalgebra::allocator::Allocator;
use rigid::Transform;
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

    fn iterate(&mut self, fixed: &Matrix<D>, moving: &Matrix<D>, probabilities: &Probabilities<D>) -> f64 {
        let np = probabilities.pt1.iter().sum::<f64>();
        let mu_fixed = fixed.transpose() * &probabilities.pt1  / np;
        let mu_moving = moving.transpose() * &probabilities.p1  / np;
        let a = probabilities.px.transpose() * moving  - np * &mu_fixed * mu_moving.transpose();
        let svd = a.svd(true, true);
        let mut c = SquareMatrix::<D>::identity();
        if !self.rigid.allow_reflections {
            c[(D::dim() - 1, D::dim() - 1)] = (svd.u.as_ref().unwrap() * svd.v_t.as_ref().unwrap()).determinant();
        }
        self.rotation = svd.u.unwrap() * &c * svd.v_t.unwrap();
        let trace = (SquareMatrix::<D>::from_diagonal(&svd.singular_values) * c).trace();
        let a = (0..D::dim()).map(|d| {
            fixed.column(d).iter().zip(probabilities.pt1.iter()).map(|(n, p)| n.powi(2) * p).sum::<f64>()
        }).sum::<f64>();
        let b = np * (mu_fixed.transpose() * &mu_fixed)[0];
        let c = (0..D::dim()).map(|d| {
            moving.column(d).iter().zip(probabilities.p1.iter()).map(|(n, p)| n.powi(2) * p).sum::<f64>()
        }).sum::<f64>();
        let d = np * (mu_moving.transpose() * &mu_moving)[0];
        let denominator = np * D::dim() as f64;
        let sigma2 = if self.rigid.scale {
            self.scale = trace / (c - d);
            ((a - b - self.scale * trace) / denominator).abs()
        } else {
            ((a - b + c - d - 2. * trace) / denominator).abs()
        };
        self.translation = mu_fixed - self.scale * &self.rotation * mu_moving;
        sigma2
    }

    fn transform(&self, moving: &Matrix<D>) -> Matrix<D> {
        let mut moved = self.scale * moving * &self.rotation;
        for d in 0..D::dim() {
            moved.column_mut(d).add_scalar_mut(self.translation[d]);
        }
        moved
    }

    fn denormalize(&mut self, normalization: &Normalization<D>) {
        self.scale *= normalization.fixed.scale / normalization.moving.scale;
        self.translation = normalization.fixed.scale * &self.translation + &normalization.fixed.offset - self.scale * &self.rotation * &normalization.moving.offset;
    }
}

impl<'a, D> From<Registration<'a, D>> for Transform<D>
where
    D: DimName,
    <D as DimName>::Value: Mul + Mul<UInt>,
    <<D as DimName>::Value as Mul>::Output: ArrayLength<f64>,
    <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
{
    fn from(registration: Registration<D>) -> Transform<D> {
        Transform {
            rotation: registration.rotation,
            scale: if registration.rigid.scale {
                Some(registration.scale)
            } else {
                None
            },
            translation: registration.translation,
        }
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
