//! Register two point sets with Coherent Point Drift (cpd).

#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unsafe_code,
        unstable_features,
        unused_import_braces, unused_qualifications)]

extern crate failure;
#[macro_use]
extern crate failure_derive;
extern crate generic_array;
extern crate nalgebra;

pub mod normalize;
pub mod rigid;
pub mod utils;
mod registration;
mod run;
mod runner;

pub use nalgebra::{U2, U3};
pub use normalize::Normalize;
pub use registration::Registration;
pub use rigid::Rigid;
pub use run::Run;
pub use runner::Runner;

/// Our custom dynamic-row matrix type.
pub type Matrix<D> = nalgebra::MatrixMN<f64, nalgebra::Dynamic, D>;

/// Our custom square matrix type.
pub type SquareMatrix<D> = nalgebra::MatrixN<f64, D>;

/// Our custom vector type.
pub type Vector<D> = nalgebra::VectorN<f64, D>;

/// Our UInt, used for matrix indexing.
pub type UInt = generic_array::typenum::UInt<
    generic_array::typenum::UTerm,
    generic_array::typenum::B1,
>;
