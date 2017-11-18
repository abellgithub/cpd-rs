use super::Rigid;

/// An error that is returned when asked to normalize independenty without scaling.
#[derive(Clone, Copy, Debug, Fail)]
#[fail(display = "Cannot use Normalize::Independent without rigid scaling")]
pub struct CannotNormalizeIndependentlyWithoutScale {}

/// A `Registration` for running rigid registrations.
#[derive(Debug)]
pub struct Registration<'a> {
    rigid: &'a Rigid,
}

impl<'a> Registration<'a> {
    /// Creates an new registration from a rigid.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::rigid::{Rigid, Registration};
    /// let rigid = Rigid::new();
    /// let registration = Registration::new(&rigid).unwrap();
    /// ```
    pub fn new(
        rigid: &'a Rigid,
    ) -> Result<Registration<'a>, CannotNormalizeIndependentlyWithoutScale> {
        if rigid.runner.requires_scaling() && !rigid.scale {
            Err(CannotNormalizeIndependentlyWithoutScale {})
        } else {
            Ok(Registration { rigid: rigid })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use {Normalize, Runner};

    #[test]
    fn normalize_independent_and_no_scale() {
        let rigid = Runner::new()
            .normalize(Normalize::Independent)
            .rigid()
            .scale(false);
        assert!(Registration::new(&rigid).is_err());
    }
}
