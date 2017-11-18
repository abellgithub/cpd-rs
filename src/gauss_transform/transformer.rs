use Matrix;
use gauss_transform::{InvalidOutlierWeight, Probabilities};
use nalgebra::DimName;

/// Runs gauss transforms on two point sets.
#[derive(Debug)]
pub struct Transformer<'a, D>
where
    D: DimName,
{
    fixed: &'a Matrix<D>,
    outlier_weight: f64,
}

impl<'a, D> Transformer<'a, D>
where
    D: DimName,
{
    /// Creates a new transformer.
    ///
    /// Returns an error if the outlier weight is not between zero and one.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::utils;
    /// use cpd::gauss_transform::Transformer;
    /// let matrix = utils::random_matrix2(10);
    /// let transformer = Transformer::new(&matrix, 0.1).unwrap();
    /// assert!(Transformer::new(&matrix, 1.1).is_err());
    /// ```
    pub fn new(
        fixed: &'a Matrix<D>,
        outlier_weight: f64,
    ) -> Result<Transformer<'a, D>, InvalidOutlierWeight> {
        if outlier_weight < 0. || outlier_weight > 1. {
            Err(InvalidOutlierWeight(outlier_weight))
        } else {
            Ok(Transformer {
                fixed: fixed,
                outlier_weight: outlier_weight,
            })
        }
    }

    /// Returns probabilities as calculated for these moving points and sigma2.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::utils;
    /// use cpd::gauss_transform::Transformer;
    /// let fixed = utils::random_matrix2(10);
    /// let transformer = Transformer::new(&fixed, 0.1).unwrap();
    /// let moving = utils::random_matrix2(10);
    /// let probabilities = transformer.probabilities(&moving, 1.0);
    /// ```
    pub fn probabilities(&self, moving: &Matrix<D>, sigma2: f64) -> Probabilities<D> {
        use nalgebra::DVector;
        use std::f64::consts::PI;

        let fixed_nrows = self.fixed.nrows() as f64;
        let moving_nrows = moving.nrows() as f64;
        let ksig = -2.0 * sigma2;
        let outliers = (self.outlier_weight * moving_nrows *
                            (-ksig * PI).powf(0.5 * D::dim() as f64)) /
            ((1. - self.outlier_weight) * fixed_nrows);

        let mut p = DVector::<f64>::zeros(moving.nrows());
        let mut p1 = DVector::<f64>::zeros(moving.nrows());
        let mut pt1 = DVector::<f64>::zeros(self.fixed.nrows());
        let mut px = Matrix::<D>::zeros(moving.nrows());
        let mut error = 0.;

        for n in 0..self.fixed.nrows() {
            let mut sp = 0.;
            for m in 0..moving.nrows() {
                let norm: f64 = self.fixed
                    .row(n)
                    .iter()
                    .zip(moving.row(m).iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                p[m] = (norm / ksig).exp();
                sp += p[m];
            }
            sp += outliers;
            pt1[n] = 1. - outliers / sp;
            p1 += &p / sp;
            for d in 0..D::dim() {
                let mut column = px.column_mut(d);
                column += (self.fixed[(n, d)] / sp) * &p;
            }
            error += -sp.ln();
        }
        error += D::dim() as f64 * fixed_nrows * sigma2.ln() / 2.;
        Probabilities {
            p1: p1,
            pt1: pt1,
            px: px,
            error: error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    use utils;

    fn dvector(slice: &[f64]) -> DVector<f64> {
        DVector::from_iterator(slice.len(), slice.iter().map(|&n| n))
    }

    #[test]
    fn invalid_outlier_weight() {
        let matrix = utils::random_matrix2(10);
        assert_eq!(
            InvalidOutlierWeight(-1.),
            Transformer::new(&matrix, -1.).unwrap_err()
        );
        assert_eq!(
            InvalidOutlierWeight(1.1),
            Transformer::new(&matrix, 1.1).unwrap_err()
        );
    }

    #[test]
    fn probabilities() {
        let fixed = utils::matrix2_from_slice(&[1., 3.]);
        let transformer = Transformer::new(&fixed, 0.1).unwrap();
        let moving = utils::matrix2_from_slice(&[2., 5., 4., 6.]);
        let probabilities = transformer.probabilities(&moving, 1.0);
        assert_relative_eq!(dvector(&[0.2085, 0.]), probabilities.p1, epsilon = 1e-4);
        assert_relative_eq!(dvector(&[0.2085]), probabilities.pt1, epsilon = 1e-4);
        assert_relative_eq!(
            utils::matrix2_from_slice(&[0.2085, 0., 0.6256, 0.]),
            probabilities.px,
            epsilon = 1e-4
        );
        assert_relative_eq!(-0.5677, probabilities.error, epsilon = 1e-4);
    }
}
