use {SquareMatrix, UInt, Vector};
use generic_array::ArrayLength;
use nalgebra::DimName;
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
