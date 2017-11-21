use {SquareMatrix, UInt, Vector};
use generic_array::ArrayLength;
use nalgebra::{DimName, Rotation3, Transform3, Translation3, U3};
use std::ops::Mul;

/// The result of a rigid transform.
#[derive(Debug)]
pub struct Transform<D>
where
    D: DimName,
    <D as DimName>::Value: Mul + Mul<UInt>,
    <<D as DimName>::Value as Mul>::Output: ArrayLength<f64>,
    <<D as DimName>::Value as Mul<UInt>>::Output: ArrayLength<f64>,
{
    /// The rotation matrix.
    pub rotation: SquareMatrix<D>,

    /// The scaling, if requested.
    pub scale: Option<f64>,

    /// The translation matrix
    pub translation: Vector<D>,
}

impl Transform<U3> {
    /// Converts a three-dimensional transform to a Transform3.
    pub fn as_transform3(&self) -> Transform3<f64> {
        use alga::general::SubsetOf;
        let rotation = Rotation3::from_matrix_unchecked(self.rotation);
        let transform: Transform3<f64> = rotation.to_superset();
        transform * Translation3::from_vector(self.translation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_transform3() {
        let transform = Transform {
            rotation: SquareMatrix::<U3>::identity(),
            scale: None,
            translation: Vector::<U3>::new(1., 2., 3.),
        };
        let transform = transform.as_transform3();
        assert_eq!(1., transform.matrix()[(0, 0)]);
        assert_eq!(1., transform.matrix()[(0, 3)]);
        assert_eq!(2., transform.matrix()[(1, 3)]);
        assert_eq!(3., transform.matrix()[(2, 3)]);
    }
}
