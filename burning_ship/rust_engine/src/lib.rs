/// Engine algorithm version.
///
/// Bump MAJOR when the encode/decode output changes for the same input.
/// Bump MINOR for new features that don't change existing outputs.
/// Pre-1.0.0: algorithm is unstable and may change freely.
pub const ENGINE_VERSION: &str = "0.1.0";

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
pub mod render_cache;
pub mod argon2_hash;
pub mod ffi;
