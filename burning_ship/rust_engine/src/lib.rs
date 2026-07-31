/// Engine algorithm version.
///
/// Bump MAJOR when the encode/decode output changes for the same input.
/// Bump MINOR for new features that don't change existing outputs.
/// Pre-1.0.0: algorithm is unstable and may change freely.
///
/// 0.2.0: raised the encode/decode island-discovery escape cap (GUI_PARAMS
///        max_iter 64 -> 1024) so deep bisection levels near the set boundary
///        stay navigable instead of starving discovery and stalling.  This is
///        an output-changing (backward-incompatible) algorithm change: 0.1.0
///        encodings do not reproduce across it, so frozen vectors built under
///        0.1.0 are STALE and are rebuilt at the stable release.
pub const ENGINE_VERSION: &str = "0.2.0";

/// Log macro that compiles to nothing unless the `verbose` feature is enabled.
/// Build with `cargo build --release --features verbose` to activate.
#[macro_export]
macro_rules! log_verbose {
    ($($arg:tt)*) => {
        #[cfg(feature = "verbose")]
        eprintln!($($arg)*);
    };
}

pub mod fixed;
pub mod fractal;
pub mod discovery;
pub mod bisect;
pub mod leaf_enum;
pub mod render_cache;
pub mod argon2_hash;
pub mod shamir;
pub mod orbit;
pub mod setup_tiers;
pub mod text;
pub mod protocol;
pub mod ffi;
