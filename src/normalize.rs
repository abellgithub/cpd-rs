/// Normalization strategies.
///
/// By normalizing the points before running registrations, error sums can be kept smaller and, in
/// some cases, prevented from overflowing.
///
/// The default normalization strategy is SameScale:
///
/// ```
/// use cpd::Normalize;
/// assert_eq!(Normalize::SameScale, Normalize::default());
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Normalize {
    /// Normalize both point sets independently.
    ///
    /// When used with a rigid registration, requires that scaling be enabled.
    Independent,

    /// Normalize both points sets with the same scale value.
    ///
    /// Useful for LiDAR data, where you might want to reduce the coordiante values but you don't
    /// want to scale the points.
    SameScale,

    /// Don't normalize the points.
    None,
}

impl Normalize {
    /// Returns true if this normalization requires scaling.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpd::Normalize;
    /// assert!(Normalize::Independent.requires_scaling());
    /// assert!(!Normalize::SameScale.requires_scaling());
    /// assert!(!Normalize::None.requires_scaling());
    /// ```
    pub fn requires_scaling(&self) -> bool {
        match *self {
            Normalize::Independent => true,
            _ => false,
        }
    }
}

impl Default for Normalize {
    fn default() -> Normalize {
        Normalize::SameScale
    }
}
