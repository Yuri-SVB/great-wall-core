//! The default landmark path of a board's leaf (great-wall-docs
//! `next-steps/landmark-path-spec.md`): integer and fixed-point arithmetic
//! only, so every device computes the same path.
//!
//! Modules, each independent and tested alone:
//! - [`raster`]: the scan raster and cell bitmaps;
//! - [`features`]: labelling, descent, features, the leaf feature;
//! - [`hull`]: the integer convex hull and its fill;
//! - [`coast`]: the outer shell and fiords;
//! - [`metric`]: the zoom distance on points `(b, s)`;
//! - [`pathfind`]: Dijkstra over a caller-supplied graph;
//! - [`path`]: the default path, built from the modules above.

pub mod coast;
pub mod features;
pub mod hull;
pub mod metric;
pub mod params;
pub mod path;
pub mod pathfind;
pub mod raster;
mod wide;

pub use features::{Feature, FeatureKind};
pub use params::{LandmarkParams, Ratio};
pub use path::{landmark_path, LandmarkPath, PathError};

/// Version of the landmark-path output. Bump it whenever the path computed
/// for the same board, leaf and parameters changes; the frozen vectors
/// (`test_vectors/landmark-v<version>/`) carry it.
pub const LANDMARK_PATH_VERSION: &str = "0.1.0";
