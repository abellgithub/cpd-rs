/// A trait for all structures that can be registered by a runner.
pub trait Registration {
    /// The struct that is returned after a registration. Holds the registration information, e.g.
    /// rotation and translation matrices.
    type Transform;
}
