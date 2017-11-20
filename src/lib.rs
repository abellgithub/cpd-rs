//! Register two point sets with Coherent Point Drift (cpd).
//!
//! Coherent Point Drifit is a point set registration algorithm created by [Andriy
//! Myroneno](https://sites.google.com/site/myronenko/research/cpd). It calculates the best
//! alignment between two point sets using one three algorithms:
//!
//! - rigid: rotation, translation, and possibly scaling
//! - nonrigid (not yet implemented by cpd-rs): nonrigid transformation goverend by motion
//! coherence theory.
//! - affine (not yet implemented by cpd-rs): an affine matrix transformation.
//!
//! This is a pure-rust implementation of cpd, relyign on [nalgebra](http://nalgebra.org/) for the
//! linear algebra.
//!
//! # Architecture
//!
//! The heavy lifting of cpd is done by [Runner::run](runner/struct.Runner.html#method.run), which
//! takes as its inputs a fixed matrix, a moving matrix, and a
//! [Registration](trait.Registration.html), and outputs information about the run such as whether
//! the algorithm converged and transformation parateters. In order to configure the runner and the
//! registration, builders are used. E.g., a runner builder:
//!
//! ```
//! let runner = cpd::Runner::new();
//! ```
//!
//! Converting a runner builder to a rigid builder:
//!
//! ```
//! let rigid = cpd::Runner::new().rigid();
//! ```
//!
//! Creating a rigid builder directly:
//!
//! ```
//! let rigid = cpd::Rigid::new();
//! ```
//!
//! And [Rigid::register](rigid/struct.Rigid.html#method.register) sets everything up and calls
//! `Runner::run`:
//!
//! ```
//! use cpd::{Rigid, utils};
//! let fixed = utils::random_matrix2(10);
//! let moving = utils::random_matrix2(10);
//! let rigid = Rigid::new();
//! let run = rigid.register(&fixed, &moving).unwrap();
//! ```

#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unsafe_code,
        unstable_features,
        unused_import_braces, unused_qualifications)]

#[cfg(test)]
#[macro_use]
extern crate approx;
extern crate failure;
#[macro_use]
extern crate failure_derive;
extern crate generic_array;
#[macro_use]
extern crate log;
extern crate nalgebra;

pub mod gauss_transform;
pub mod normalize;
pub mod rigid;
pub mod runner;
pub mod utils;
mod registration;

pub use nalgebra::{U2, U3};
pub use normalize::{Normalization, Normalize};
pub use registration::Registration;
pub use rigid::Rigid;
pub use runner::{Run, Runner};

/// Our custom dynamic-row matrix type.
pub type Matrix<D> = nalgebra::MatrixMN<f64, nalgebra::Dynamic, D>;

/// Our custom square matrix type.
pub type SquareMatrix<D> = nalgebra::MatrixN<f64, D>;

/// Our custom vector type.
pub type Vector<D> = nalgebra::VectorN<f64, D>;

/// Our custom row vector type.
pub type RowVector<D> = nalgebra::RowVectorN<f64, D>;

/// Our UInt, used for matrix indexing.
pub type UInt = generic_array::typenum::UInt<
    generic_array::typenum::UTerm,
    generic_array::typenum::B1,
>;
