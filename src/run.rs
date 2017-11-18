/// The result of a cpd run.
#[derive(Clone, Copy, Debug)]
pub struct Run {
    /// Did this run converge?
    pub converged: bool,
}
