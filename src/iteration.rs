use Matrix;
use nalgebra::DimName;

/// The result of one iteration of cpd.
#[derive(Debug)]
pub struct Iteration<D>
where
    D: DimName,
{
    /// The moving points after this iteration.
    pub moved: Matrix<D>,

    /// The calculated sigma2 (i.e. bandwidth, i.e. how far afield the algorithm should look for
    /// correspondences).
    pub sigma2: f64,
}
