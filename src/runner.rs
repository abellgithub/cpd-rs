use Rigid;

/// Generic interface for running cpd registration methods.
///
/// Use the builder pattern to configure how cpd registrations are run.
///
/// ```
/// use cpd::Runner;
/// let runner = Runner::new().max_iterations(100); // etc
/// ```
///
/// Use methods like `rigid()` to specifiy the type of registration, and convert the builder into a
/// method-specific builder:
///
/// ```
/// use cpd::Runner;
/// let runner = Runner::new().rigid();
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Runner {
    max_iterations: usize,
}

impl Runner {
    /// Creates a new, default runner.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::Runner;
    /// let runner = Runner::new();
    /// ```
    pub fn new() -> Runner {
        Runner::default()
    }

    /// Sets the maximum number of iterations when running cpd.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::Runner;
    /// let runner = Runner::new().max_iterations(100);
    /// ```
    pub fn max_iterations(mut self, max_iterations: usize) -> Runner {
        self.max_iterations = max_iterations;
        self
    }

    /// Returns a rigid registration builder that will use this runner.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::Runner;
    /// let rigid = Runner::new().rigid();
    /// ```
    pub fn rigid(self) -> Rigid {
        self.into()
    }
}
